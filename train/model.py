"""PPO 的 Actor-Critic 网络：3 层 3x3 CNN + 1x1 压缩 + 共享全连接 + 三个头。

动作空间是**因子化**的：方向头 5 类（上下左右 + idle）× 放泡头 2 类（放 / 不放），
两个头条件独立地从同一个 trunk 出来。不做成 6 个互斥动作是规则决定的 ——
炸弹人最基本的操作就是"边跑边放"，扁平动作空间会强迫你在放泡那一 tick 站住。
联合对数概率是两个头之和，熵也是两个头之和。

**输入是 env 级共享的观测**（见 `sim/config.py` 文件头）。"我是谁"通过
`forward(obs, pid)` 表达：视角只是存储张量的一个通道置换，而置换输入通道
**等价于置换第一层卷积的权重**：

    sum_j view[j] * w[:, j]  =  sum_k shared[k] * w[:, inv[k]]      inv = argsort(perm)

所以只要把 (16, C, 3, 3) 这个小权重按 `inv_perm[pid]` 索引一下就行，
(N, C, H, W) 的观测数据一个字节都不用搬。`tests/test_train.py::
test_weight_perm_equals_data_gather` 钉住了这个等价性 —— 有那条测试在，
这个技巧就不是"聪明代码"，改坏了会立刻红。

刻意保持小（11x11 地图约 21 万参数）：RL 的瓶颈是样本效率而不是模型容量，
小网络在同样墙上时间里能采到更多经验。地图放大到 15x15 以上、或者卡在
瓶颈很久之后，再考虑加宽 / 上 ResNet。

归一化用 LayerNorm 而不是 BatchNorm：RL 的 batch 数据高度相关且分布随策略
漂移，BatchNorm 的running stats 会失真，LayerNorm 与 batch 大小无关。

输出的是 **logits 而不是 softmax 概率**——掩码要在 logits 上加 -inf，
在概率上乘 0 再归一化会在全掩码时产生 NaN。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sim.config import N_BOMB, N_MOVES, n_obs_channels, view_perm

# LSTM 架构的输入特征维度（与 sim/obs.local_view_features 的约定一致）
REL_FEAT = 60        # max_t=10 目标 × (dx, dy, type_onehot[4]) = 6 维
GLOB_FEAT = 5        # t 进度 / 自己 hp / 脚下 danger / 场上炸弹数 / fuse 均值
LSTM_HIDDEN = 128


def infer_players(c: int) -> int:
    """从通道数反推玩家数。支持两种布局：
    旧 (2P+3)、带扩展观测 obs_extra=1+3P (5P+4，rw7/rw8 同一布局)。
    7 通道对 5P+4 无整数解，不会误判；优先按 5P+4 反推，其余按 2P+3 兜底。
    """
    if (c - 4) % 5 == 0:
        return (c - 4) // 5
    return (c - 3) // 2


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        n_moves: int = N_MOVES,
        n_bomb: int = N_BOMB,
        arch: str = "cnn",
        n_players: int | None = None,
    ) -> None:
        super().__init__()
        c, h, w = obs_shape
        self.obs_shape = obs_shape
        self.n_moves, self.n_bomb = n_moves, n_bomb
        self.arch = arch
        assert arch in ("cnn", "mlp", "lstm")
        # 通道数 = 2P+3（视角置换部分）+ 可选 obs_extra(P)（世界信息尾部原样保留）。
        # obs_extra = 1+3P（宝箱/无敌/可用泡/上限，c=5P+4）；旧档 7 通道 = 2P+3。
        if n_players is None:
            n_players = infer_players(c)
        base = 2 * n_players + 3
        assert c in (base, n_obs_channels(n_players)), \
            f"通道数 {c} 必须 = {base}（旧）或 = {n_obs_channels(n_players)}（extra）"
        self.n_players = n_players
        if arch == "cnn":
            # 第一层单独拿出来：forward 时要按视角索引它的输入通道维
            self.conv0 = nn.Conv2d(c, 16, 3, padding=1)
            self.conv = nn.Sequential(
                nn.LayerNorm([16, h, w]),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.LayerNorm([32, h, w]),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.LayerNorm([64, h, w]),
                nn.ReLU(inplace=True),
                # 1x1 卷积做通道压缩：直接 flatten 64 通道会让第一个全连接层
                # 吃掉 200 万参数（64×11×11×256），压到 8 通道后降到 ~12 万，
                # 而空间分辨率一格没丢 —— 躲泡泡靠的正是格子级的精度。
                nn.Conv2d(64, 8, 1),
                nn.LayerNorm([8, h, w]),
                nn.ReLU(inplace=True),
            )
            # inv_perm[pid] 直接喂给 weight 的通道索引，见文件头的推导
            inv = torch.zeros((self.n_players, c), dtype=torch.long)
            for pid in range(self.n_players):
                perm = torch.tensor(view_perm(pid, self.n_players, c))
                inv[pid] = torch.argsort(perm)
            self.register_buffer("inv_perm", inv, persistent=False)
            flat = 8 * h * w
        elif arch == "lstm":
            # LSTM 架构：局部 7×7 视野 CNN + 相对坐标 MLP + 全局状态 MLP +
            # 融合 + LSTM 时序（用户给的 BombermanNet 结构，动作空间保留 5+2 双头）。
            # 输入不再是共享全图 (N,C,H,W)，而是每角色的特征三元组：
            #   local (N, C, 7, 7) 以自己为中心的窗口，rel (N, max_t, 6) 目标
            #   相对坐标，glob (N, G) 全局标量 —— 由 sim/obs.local_view_features() 生成。
            # 视角信息已编码进局部坐标，不需要 inv_perm/inv_cols 权重置换。
            self.l_conv1 = nn.Conv2d(c, 32, 3, padding=1)       # 7x7 -> 7x7
            self.l_conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.l_pool = nn.MaxPool2d(2)                        # 7x7 -> 3x3
            self.l_cnn_fc = nn.Linear(64 * 3 * 3, 128)
            self.l_rel_fc1 = nn.Linear(REL_FEAT, 64)             # REL_FEAT = max_t*6
            self.l_rel_fc2 = nn.Linear(64, 32)
            self.l_glob_fc1 = nn.Linear(GLOB_FEAT, 16)
            self.l_glob_fc2 = nn.Linear(16, 8)
            self.l_fusion = nn.Linear(128 + 32 + 8, 256)
            self.lstm = nn.LSTM(256, LSTM_HIDDEN, 1, batch_first=True)
            flat = LSTM_HIDDEN                       # 双头输入维度 = LSTM 输出
        else:
            # MLP：全局感受野。观测是 (B,C,H,W) 共享张量，flatten 成 (B, C*H*W)
            # 是固定线性序（C 主序），视角 = 第一层 Linear 权重**按通道块重排列**，
            # 与 CNN 的 conv0 权重通道索引是同一套技巧 —— 数据一个字节不搬。
            # 危险图通道已把"火会烧到哪"算好画出来，CNN 的局部归纳偏置收益
            # 有限，MLP 参数更少（280k→192k）且在 DCU/GPU 上 GEMM 比小卷积快。
            inv_cols = torch.zeros((self.n_players, c * h * w), dtype=torch.long)
            for pid in range(self.n_players):
                perm = torch.tensor(view_perm(pid, self.n_players, c))
                inv = torch.argsort(perm)         # 存储通道 k 在视角中的位置
                for k in range(c):
                    # 权重列块重排：新权重第 k 块 ← 原权重第 inv[k] 块。
                    # （视角通道 inv[k] 存的是存储通道 k —— 与 CNN 的
                    #   conv0.weight[:, inv_perm] 同一套推导，见文件头。）
                    src = torch.arange(h * w, dtype=torch.long) + int(inv[k]) * (h * w)
                    dst = k * (h * w)
                    inv_cols[pid, dst:dst + h * w] = src
            self.register_buffer("inv_cols", inv_cols, persistent=False)
            flat = c * h * w
        self.shared = None if arch == "lstm" else nn.Sequential(
            nn.Linear(flat, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
        )
        self.move_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True),
                                       nn.Linear(64, n_moves))
        self.bomb_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True),
                                       nn.Linear(64, n_bomb))
        self.critic = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True),
                                    nn.Linear(64, 1))
        self.apply(self._init)
        # 策略头最后一层用小增益初始化：开局动作分布接近均匀，探索更充分
        for head in (self.move_head, self.bomb_head):
            nn.init.orthogonal_(head[-1].weight, gain=0.01)
            nn.init.zeros_(head[-1].bias)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight, gain=2**0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, obs, pid: int = 0, hidden=None):
        """→ (move_logits, bomb_logits, value[, new_hidden])。

        cnn/mlp：obs 是 (B, C, H, W) **共享**观测。`pid` 决定用哪个视角读这份
        共享张量。数据不动，只索引第一层权重（CNN：conv0 输入通道；MLP：
        shared 第一层 Linear 的列块）。obs 可以是 fp16（模拟器默认这么存）；
        这里 cast 一次到参数 dtype。端到端想省掉这次 cast，就把整段 forward
        放进 `torch.autocast`。

        lstm：obs 是特征三元组 (local, rel, glob)（见 sim/obs.local_view_features），
        pid 忽略（局部坐标已编码视角），hidden 是 LSTM 的 (h, c) 元组（None =
        零状态）。返回 4 元组，最后一个是新 hidden —— 调用方按 arch 解包。
        """
        if self.arch == "lstm":
            local, rel, glob = obs
            fused = self.extract_fused(local, rel, glob)
            lo, nh = self.lstm(fused.unsqueeze(1), hidden)
            lo = lo.squeeze(1)
            return (self.move_head(lo), self.bomb_head(lo),
                    self.critic(lo).squeeze(-1), nh)
        if self.arch == "cnn":
            w = self.conv0.weight[:, self.inv_perm[pid]]
            x = F.conv2d(obs.to(w.dtype), w, self.conv0.bias, padding=1)
            x = self.shared(self.conv(x).flatten(1))
        else:
            x = obs.to(self.shared[0].weight.dtype).reshape(obs.shape[0], -1)
            w0 = self.shared[0].weight[:, self.inv_cols[pid]]   # 视角 = 列块重排
            x = F.linear(x, w0, self.shared[0].bias)
            x = self.shared[1:](x)
        return self.move_head(x), self.bomb_head(x), self.critic(x).squeeze(-1)

    def extract_fused(self, local, rel, glob) -> torch.Tensor:
        """LSTM 架构：conv/rel/glob/fusion 逐帧独立部分 → (B, 256) 融合特征。

        BPTT 优化：时序只存在于 LSTM 层，前面的特征提取可以一次喂整个
        (T*N, ...) 大 batch，而不是每 tick 一小批 —— 910B 上大 batch 的
        GEMM/conv 效率高得多。collect 每 tick 只 1 帧，也走这里（batch=N）。
        """
        x = torch.relu(self.l_conv1(local.to(self.l_conv1.weight.dtype)))
        x = self.l_pool(torch.relu(self.l_conv2(x)))
        x = torch.relu(self.l_cnn_fc(x.view(x.size(0), -1)))
        r = torch.relu(self.l_rel_fc1(
            rel.to(self.l_rel_fc1.weight.dtype).view(rel.size(0), -1)))
        r = torch.relu(self.l_rel_fc2(r))
        g = torch.relu(self.l_glob_fc1(glob.to(self.l_glob_fc1.weight.dtype)))
        g = torch.relu(self.l_glob_fc2(g))
        return torch.relu(self.l_fusion(torch.cat([x, r, g], dim=1)))

    @staticmethod
    def masked_dist(logits: torch.Tensor, mask: torch.Tensor):
        """在 logits 上施加合法动作掩码，返回 Categorical 分布。"""
        neg_inf = torch.finfo(logits.dtype).min
        masked = torch.where(mask, logits, torch.full_like(logits, neg_inf))
        return torch.distributions.Categorical(logits=masked)

    def dists(self, obs, move_mask, bomb_mask, pid: int = 0, hidden=None):
        if self.arch == "lstm":
            ml, bl, val, nh = self(obs, pid, hidden)
            return (self.masked_dist(ml, move_mask), self.masked_dist(bl, bomb_mask),
                    val, nh)
        ml, bl, val = self(obs, pid)
        return (self.masked_dist(ml, move_mask), self.masked_dist(bl, bomb_mask), val)

    def act(self, obs, move_mask, bomb_mask, pid: int = 0, hidden=None):
        """采样一步。返回 (actions (B,2), logp (B,), value (B,)[, new_hidden])。

        保持 Categorical.sample —— 实测 Gumbel-max 替代在 HIP 上慢 3 倍
        （log_softmax 处理 -inf + 更多 kernel），且 profiler 的 Memcpy DtoH
        是测量开销（CUDA 事件读取）非真实同步（2026-08-10 对比后回滚）。
        """
        if self.arch == "lstm":
            dm, db, val, nh = self.dists(obs, move_mask, bomb_mask, pid, hidden)
            am, ab = dm.sample(), db.sample()
            logp = dm.log_prob(am) + db.log_prob(ab)
            return torch.stack([am, ab], dim=-1), logp, val, nh
        dm, db, val = self.dists(obs, move_mask, bomb_mask, pid)
        am, ab = dm.sample(), db.sample()
        logp = dm.log_prob(am) + db.log_prob(ab)
        return torch.stack([am, ab], dim=-1), logp, val

    def evaluate(self, obs, move_mask, bomb_mask, actions, pid: int = 0,
                 hidden=None):
        """重算旧动作的 (logp, entropy, value)，PPO 更新用。"""
        if self.arch == "lstm":
            dm, db, val, _ = self.dists(obs, move_mask, bomb_mask, pid, hidden)
            logp = dm.log_prob(actions[..., 0]) + db.log_prob(actions[..., 1])
            return logp, dm.entropy() + db.entropy(), val
        dm, db, val = self.dists(obs, move_mask, bomb_mask, pid)
        logp = dm.log_prob(actions[..., 0]) + db.log_prob(actions[..., 1])
        return logp, dm.entropy() + db.entropy(), val

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
