// bomber_kernels.cu —— 批量泡泡堂"基础关卡"的 GPU 算子
//
// 设计要点（这是本项目想展示的核心）：
//
// 1) 并行度来自"关卡"，不来自"玩家"。单关卡只有 2~4 个角色，为它们各开一个
//    线程，内核启动开销比计算本身还大。所以 step 内核是 **一线程一关卡**，
//    关卡内部的逻辑照常串行写，可读性和 CPU 版一致。
//
// 2) 布局是 SoA 且 **env 维放最内层**：idx(cell, env) = cell * num_envs + env。
//    同一 warp 里 32 个线程是 32 个相邻关卡，访问同一个 cell → 完全合并访存。
//    如果按 AoS（每个关卡一块连续内存）来存，同一 warp 会跨 H*W 个元素跳，
//    带宽直接掉一个数量级。
//
// 3) 连锁爆炸不用队列递归，改成 **固定轮数的同步迭代**：每轮只读上一轮的
//    covered，只写本轮的 covered，写冲突被彻底消除，也不需要 atomic。
//    轮数上限 max_chain，与参考实现完全一致（见 RULES.md 第 4 条）。
//
// 4) 危险度图（观测通道）用 **gather 而不是 scatter**：每个格子自己朝四个方向
//    往外看 blast 格，遇墙即停，取遇到的泡泡权重最大值。这样完全无需 atomicMax
//    和临时缓冲，且天然是每线程独立的纯读操作。
//
// 5) 位置是 **连续坐标**（float y/x），每 tick 前进 speed/tick_hz 格。碰撞是
//    轴对齐盒 + 单轴滑动；因为 radius<0.5 且单 tick 位移远小于格宽，每个轴
//    只需要测 2 个格子，既不用遍历邻域也不用 substep。观测里的位置通道是
//    双线性铺开，亚格偏移不会被量化掉。
//
// 6) 动作是因子化的 (move, bomb)：move ∈ [0,4]（四方向 + idle），bomb ∈ {0,1}。
//    "边跑边放"是炸弹人的基本操作，扁平动作空间做不到。

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>

#define MAX_P 4
#define MOVE_IDLE 4
#define N_MOVES 5
#define N_BOMB 2
#define COLLIDE_EPS 1e-4f

namespace {

__device__ __forceinline__ int idx2(int cell, int env, int num_envs) {
  return cell * num_envs + env;
}

// 四个方向，索引与 Python 侧 config.DIRS / 方向头编码一致
__device__ __constant__ int kDRow[4] = {-1, 1, 0, 0};
__device__ __constant__ int kDCol[4] = {0, 0, -1, 1};

struct Geom {
  int h, w, p, num_envs;
  int fuse_max, blast, max_bombs, max_steps, max_chain;
  float step_penalty;
  float radius, step_len;
  float hit_reward, win_bonus;
  float danger_penalty, passivity_penalty;
  int passivity_ticks;
  float max_hp;
  int win_hp_scaled;
};

// 格 (row,col) 对这个角色是否不可通行。越界算不可通行；
// 碰撞盒当前已经压住该格则放行 —— 这就是"走出自己脚下刚放的泡"的无状态实现。
__device__ __forceinline__ bool impassable(int row, int col, float y, float x,
                                           const unsigned char* wall,
                                           const int* fuse, int env, const Geom g) {
  if (row < 0 || row >= g.h || col < 0 || col >= g.w) return true;
  const int i = idx2(row * g.w + col, env, g.num_envs);
  if (!(wall[i] || fuse[i] > 0)) return false;
  const float rad = g.radius;
  const int r0 = (int)floorf(y - rad), r1 = (int)floorf(y + rad);
  const int c0 = (int)floorf(x - rad), c1 = (int)floorf(x + rad);
  const bool inside = (row >= r0 && row <= r1 && col >= c0 && col <= c1);
  return !inside;
}

// 沿单轴消解碰撞：撞上就贴着障碍物停下（滑动），而不是整步作废。
__device__ __forceinline__ float resolve_axis(float coord, float delta, float other,
                                              float y, float x,
                                              const unsigned char* wall,
                                              const int* fuse, int env,
                                              const Geom g, bool vertical) {
  const float rad = g.radius;
  const float sgn = delta > 0.f ? 1.f : (delta < 0.f ? -1.f : 0.f);
  const int lead = (int)floorf(coord + sgn * rad);
  const int s0 = (int)floorf(other - rad);
  const int s1 = (int)floorf(other + rad);
  bool hit;
  if (vertical) {
    hit = impassable(lead, s0, y, x, wall, fuse, env, g) ||
          impassable(lead, s1, y, x, wall, fuse, env, g);
  } else {
    hit = impassable(s0, lead, y, x, wall, fuse, env, g) ||
          impassable(s1, lead, y, x, wall, fuse, env, g);
  }
  if (!hit) return coord;
  return sgn > 0.f ? (float)lead - rad - COLLIDE_EPS
                   : (float)lead + 1.f + rad + COLLIDE_EPS;
}

// 双线性铺开的权重：格 (row,col) 从中心在 (py,px) 的角色身上分到多少质量。
// 求和顺序与参考实现的四次 scatter_add_ 一致，float 加法顺序不能改。
__device__ __forceinline__ float splat_weight(float py, float px, int row, int col,
                                              const Geom g) {
  const float fy = fminf(fmaxf(py - 0.5f, 0.f), (float)(g.h - 1));
  const float fx = fminf(fmaxf(px - 0.5f, 0.f), (float)(g.w - 1));
  const int y0 = min(max((int)floorf(fy), 0), g.h - 1);
  const int x0 = min(max((int)floorf(fx), 0), g.w - 1);
  const int ys[2] = {y0, min(y0 + 1, g.h - 1)};
  const int xs[2] = {x0, min(x0 + 1, g.w - 1)};
  const float wy = fminf(fmaxf(fy - (float)y0, 0.f), 1.f);
  const float wx = fminf(fmaxf(fx - (float)x0, 0.f), 1.f);
  const float wys[2] = {1.f - wy, wy};
  const float wxs[2] = {1.f - wx, wx};
  float acc = 0.f;
  for (int a = 0; a < 2; ++a)
    for (int b = 0; b < 2; ++b)
      if (ys[a] == row && xs[b] == col) acc += wys[a] * wxs[b];
  return acc;
}

// 从 (row, col) 出发投射十字火焰，写入 covered。墙挡火且自身不被覆盖；
// **泡也挡火**（覆盖它但不穿透，与 blast.py::rays 同规则，连锁由
// "被点燃的泡成为新爆源"自然推进，不需要穿透实现）。
__device__ void cast_rays(int env, int row, int col, const unsigned char* wall,
                          const int* fuse, unsigned char* covered, const Geom g) {
  const int center = row * g.w + col;
  if (wall[idx2(center, env, g.num_envs)]) return;
  covered[idx2(center, env, g.num_envs)] = 1;
  for (int d = 0; d < 4; ++d) {
    for (int r = 1; r <= g.blast; ++r) {
      int nr = row + kDRow[d] * r;
      int nc = col + kDCol[d] * r;
      if (nr < 0 || nr >= g.h || nc < 0 || nc >= g.w) break;
      int cell = nr * g.w + nc;
      int i = idx2(cell, env, g.num_envs);
      if (wall[i]) break;             // 墙挡火，墙自身不烧
      covered[i] = 1;
      if (fuse[i] > 0) break;         // 泡挡火：覆盖它，但火焰不再穿透
    }
  }
}

}  // namespace

// ============================ step：一线程一关卡 ============================
__global__ void step_kernel(const unsigned char* __restrict__ wall,
                            int* __restrict__ fuse, signed char* __restrict__ owner,
                            float* __restrict__ pos, unsigned char* __restrict__ alive,
                            unsigned char* __restrict__ hp,
                            int* __restrict__ since_bomb,
                            int* __restrict__ t, const int* __restrict__ actions,
                            float* __restrict__ reward, unsigned char* __restrict__ done,
                            unsigned char* __restrict__ covered,
                            unsigned char* __restrict__ trig,
                            unsigned char* __restrict__ expanded, const Geom g) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= g.num_envs) return;
  const int ncell = g.h * g.w;
  const int N = g.num_envs;

  unsigned char alive0[MAX_P];
  unsigned char hp0[MAX_P];
  float py[MAX_P], px[MAX_P];
  for (int p = 0; p < g.p; ++p) {
    alive0[p] = alive[idx2(p, env, N)];
    hp0[p] = hp[idx2(p, env, N)];
    py[p] = pos[idx2(p * 2 + 0, env, N)];
    px[p] = pos[idx2(p * 2 + 1, env, N)];
  }

  // --- 1. 引信递减 ---
  for (int c = 0; c < ncell; ++c) {
    int i = idx2(c, env, N);
    if (fuse[i] > 0) fuse[i] -= 1;
  }

  // --- 2. 放泡（角色编号升序，与参考实现一致）。落点是移动前的中心格 ---
  bool placed[MAX_P];
  for (int p = 0; p < g.p; ++p) placed[p] = false;
  for (int p = 0; p < g.p; ++p) {
    if (!alive0[p] || actions[idx2(p * 2 + 1, env, N)] != 1) continue;
    int live = 0;
    for (int c = 0; c < ncell; ++c) {
      int i = idx2(c, env, N);
      if (owner[i] == p && fuse[i] > 0) ++live;
    }
    int cr = (int)floorf(py[p]), cc = (int)floorf(px[p]);
    int here = idx2(cr * g.w + cc, env, N);
    if (fuse[here] > 0 || live >= g.max_bombs) continue;
    fuse[here] = g.fuse_max;
    owner[here] = (signed char)p;
    placed[p] = true;
  }
  // 被动计时（与参考实现同公式）：没放泡的 +1 tick，放成功的清零
  for (int p = 0; p < g.p; ++p) {
    since_bomb[idx2(p, env, N)] = placed[p] ? 0 : since_bomb[idx2(p, env, N)] + 1;
  }

  // --- 3. 连续移动：匀速 step_len 格，逐轴滑动碰撞。角色之间不碰撞 ---
  for (int p = 0; p < g.p; ++p) {
    int a = actions[idx2(p * 2 + 0, env, N)];
    if (!alive0[p] || a < 0 || a >= MOVE_IDLE) continue;
    float dy = (float)kDRow[a] * g.step_len;
    float dx = (float)kDCol[a] * g.step_len;
    const float oy = py[p], ox = px[p];   // 两个轴都用移动前的坐标判定
    if (dy != 0.f) {
      py[p] = resolve_axis(oy + dy, dy, ox, oy, ox, wall, fuse, env, g, true);
    }
    if (dx != 0.f) {
      px[p] = resolve_axis(ox + dx, dx, oy, oy, ox, wall, fuse, env, g, false);
    }
    // 防御性边界夹紧（与 sim/move.py 同款）：坐标保持在 [rad, h-rad]，即碰撞盒
    // 最贴边但不出界。防的是 stop_pos 在边界格滑动时算出 `lead ± 1 + rad + EPS`
    // 溢出地图（角色穿出界面）。不能用格中心 [0.5, h-0.5]，那会让合法贴边姿势
    // 被钳掉，掩码与实际移动不一致。
    py[p] = fminf(fmaxf(py[p], g.radius), (float)g.h - g.radius);
    px[p] = fminf(fmaxf(px[p], g.radius), (float)g.w - g.radius);
    pos[idx2(p * 2 + 0, env, N)] = py[p];
    pos[idx2(p * 2 + 1, env, N)] = px[p];
  }

  // --- 4. 爆炸与连锁：固定轮数同步迭代，无 atomic、无队列 ---
  for (int c = 0; c < ncell; ++c) {
    int i = idx2(c, env, N);
    covered[i] = 0;
    expanded[i] = 0;
    trig[i] = (fuse[i] == 0 && owner[i] >= 0) ? 1 : 0;
  }
  for (int it = 0; it < g.max_chain; ++it) {
    bool expanded_any = false;
    for (int c = 0; c < ncell; ++c) {
      int i = idx2(c, env, N);
      if (!trig[i] || expanded[i]) continue;
      expanded[i] = 1;
      cast_rays(env, c / g.w, c % g.w, wall, fuse, covered, g);
      expanded_any = true;
    }
    if (!expanded_any) break;
    bool newly = false;
    for (int c = 0; c < ncell; ++c) {
      int i = idx2(c, env, N);
      if (fuse[i] > 0 && covered[i] && !trig[i]) {
        trig[i] = 1;
        newly = true;
      }
    }
    if (!newly) break;
  }

  // --- 5. 伤害判定：只看**中心格**（命中盒故意比碰撞盒小，见 move.py）。
  //    着火扣 1 血，血归 0 才死 —— 与参考实现 sim/torch_sim.py 同公式。
  int n_alive = 0;
  float rew[MAX_P];
  float dmg[MAX_P];
  for (int p = 0; p < g.p; ++p) {
    rew[p] = alive0[p] ? -g.step_penalty : 0.0f;
    int cr = (int)floorf(py[p]), cc = (int)floorf(px[p]);
    bool hit = alive0[p] && covered[idx2(cr * g.w + cc, env, N)] != 0;
    unsigned char h1 = hp0[p];
    dmg[p] = 0.f;
    if (hit && h1 > 0) { h1 = (unsigned char)(h1 - 1); dmg[p] = 1.f; }
    bool died = hit && h1 == 0;
    hp[idx2(p, env, N)] = h1;
    unsigned char a1 = (alive0[p] && !died) ? 1 : 0;
    alive[idx2(p, env, N)] = a1;
    n_alive += a1;
  }

  // --- 6. 清场，泡泡额度随 owner 置 -1 归还 ---
  for (int c = 0; c < ncell; ++c) {
    int i = idx2(c, env, N);
    if (trig[i]) {
      fuse[i] = 0;
      owner[i] = -1;
    }
  }

  // --- 7. 计步与终局 ---
  int tt = t[env] + 1;
  t[env] = tt;
  unsigned char d = (n_alive <= 1 || tt >= g.max_steps) ? 1 : 0;
  done[env] = d;
  // 终局胜负（与参考实现同规则）：
  //   n_alive==1 → 唯一存活着 +win_bonus / 死者 -win_bonus
  //   n_alive==P（超时全员存活）→ 血多者 +win_bonus / 血少者 -win_bonus / 血平 0
  //   n_alive==0（同时死光）→ 平局 0
  float dealt_total = 0.f;
  for (int p = 0; p < g.p; ++p) dealt_total += dmg[p];
  for (int p = 0; p < g.p; ++p) {
    rew[p] += g.hit_reward * (dealt_total - dmg[p]) - g.hit_reward * dmg[p];
    // 危险区站桩罚：站在"被在场泡泡爆炸范围覆盖"的格每 tick 扣分，
    // 大小 × danger值（1-(fuse-1)/FUSE）。与参考实现同公式：中心格向外
    // gather，遇墙/泡即停 —— 和 observe_kernel 的危险度计算同一套 gather。
    if (alive0[p]) {
      int cr = (int)floorf(py[p]), cc = (int)floorf(px[p]);
      const int ci = idx2(cr * g.w + cc, env, N);
      const int f0 = fuse[ci];
      float standing = 0.0f;
      if (wall[ci] == 0) {
        if (f0 > 0) standing = 1.0f - (float)(f0 - 1) / (float)g.fuse_max;
        for (int dir = 0; dir < 4; ++dir) {
          for (int r = 1; r <= g.blast; ++r) {
            int nr = cr + kDRow[dir] * r;
            int nc = cc + kDCol[dir] * r;
            if (nr < 0 || nr >= g.h || nc < 0 || nc >= g.w) break;
            int i = idx2(nr * g.w + nc, env, N);
            if (wall[i]) break;
            int f = fuse[i];
            if (f > 0) {
              float wgt = 1.0f - (float)(f - 1) / (float)g.fuse_max;
              standing = fmaxf(standing, wgt);
              break;                             // 泡挡火：看见它即止步
            }
          }
        }
      }
      rew[p] -= g.danger_penalty * standing;
      // 久不放炮罚（与参考实现同公式）：**有泡泡预算（还能放）**且超时没放才扣；
      // 放满了（在场泡数达到 max_bombs）不扣 —— 只有消极摆烂被罚。
      if (since_bomb[idx2(p, env, N)] >= g.passivity_ticks) {
        int live = 0;
        for (int c = 0; c < ncell; ++c) {
          int i = idx2(c, env, N);
          if (owner[i] == p && fuse[i] > 0) ++live;
        }
        if (live < g.max_bombs) rew[p] -= g.passivity_penalty;
      }
    }
    if (d) {
      if (g.win_hp_scaled) {
        // 终局 reward 按剩余血量比例（与参考实现同公式）：
        //   +win_bonus/max_hp × (自己剩余血 − 对手平均剩余血)。
        // 干净击杀拿满、残血险胜按比例少拿 —— 反"拿血换命"的 reward-hacking。
        // 双亡（hp 全 0）→ diff 0 = 平局；n_alive==1 死者 hp=0 → 负分按幸存者血量。
        float opp_sum = 0.f;
        for (int q = 0; q < g.p; ++q) {
          if (q == p) continue;
          opp_sum += (float)hp[idx2(q, env, N)];
        }
        float opp_avg = opp_sum / (float)(g.p - 1);
        float diff = (float)hp[idx2(p, env, N)] - opp_avg;
        rew[p] += (g.win_bonus / g.max_hp) * diff;
      } else if (n_alive == 1) {
        if (alive[idx2(p, env, N)]) rew[p] += g.win_bonus;
        else rew[p] -= g.win_bonus;
      } else if (n_alive == g.p) {        // 超时全员存活：血多者胜
        const int hpi = hp[idx2(p, env, N)];
        bool wins = true, loses = false;
        for (int q = 0; q < g.p; ++q) {
          if (q == p) continue;
          const int hq = hp[idx2(q, env, N)];
          if (hpi <= hq) wins = false;
          if (hpi < hq) loses = true;
        }
        if (wins) rew[p] += g.win_bonus;
        else if (loses) rew[p] -= g.win_bonus;
      }
    }
    reward[idx2(p, env, N)] = rew[p];
  }
}

// ===================== observe：一 block 一关卡，一线程一格 =====================
// 写出 (N, 2P+3, H, W)：**整个 env 共享一份**（布局见 sim/config.py 的 OBS_LAYOUT）。
// 所有通道都与"我是谁"无关，角色视角只是一个通道置换，由网络第一层的权重索引
// 吸收（train/model.py）。所以这里不再乘 P —— 写入量直接掉 P 倍，
// 而 observe 的写入正是模拟器的第一瓶颈。
//
// 相邻线程写相邻 cell → 写合并；读 SoA 状态时是跨 env 的 stride 访问 → 读不合并。
// 这是 SoA 布局对 NCHW 输出的固有代价，属于可调项（见 README "布局权衡"）。
//
// 模板参数 T 是存储 dtype：默认 at::Half（fp16），中间量一律 fp32 算，
// 只在写出那一刻 cast —— 和参考实现 sim/obs.py 的顺序一致，parity 才成立。
template <typename T>
__global__ void observe_kernel(const unsigned char* __restrict__ wall,
                               const int* __restrict__ fuse,
                               const signed char* __restrict__ owner,
                               const float* __restrict__ pos,
                               const unsigned char* __restrict__ alive,
                               const int* __restrict__ t,
                               T* __restrict__ obs, const Geom g, int n_ch) {
  const int env = blockIdx.x;
  const int ncell = g.h * g.w;
  const int N = g.num_envs;
  const float progress = (float)t[env] / (float)g.max_steps;

  float py[MAX_P], px[MAX_P];
  unsigned char al[MAX_P];
  for (int p = 0; p < g.p; ++p) {
    py[p] = pos[idx2(p * 2 + 0, env, N)];
    px[p] = pos[idx2(p * 2 + 1, env, N)];
    al[p] = alive[idx2(p, env, N)];
  }
  T* base = obs + (size_t)env * n_ch * ncell;

  for (int cell = threadIdx.x; cell < ncell; cell += blockDim.x) {
    const int row = cell / g.w, col = cell % g.w;
    const int self_i = idx2(cell, env, N);
    const bool passable = wall[self_i] == 0;
    const int f0 = fuse[self_i];
    const int o0 = owner[self_i];

    // 危险度：gather 版本 —— 自己朝四个方向往外看，遇墙即停；**遇泡也停**
    // （泡挡火，与 blast.py::danger_map 同规则），取最大权重
    float danger = 0.0f;
    if (passable) {
      if (f0 > 0) danger = 1.0f - (float)(f0 - 1) / (float)g.fuse_max;
      for (int dir = 0; dir < 4; ++dir) {
        for (int r = 1; r <= g.blast; ++r) {
          int nr = row + kDRow[dir] * r;
          int nc = col + kDCol[dir] * r;
          if (nr < 0 || nr >= g.h || nc < 0 || nc >= g.w) break;
          int i = idx2(nr * g.w + nc, env, N);
          if (wall[i]) break;
          int f = fuse[i];
          if (f > 0) {
            float wgt = 1.0f - (float)(f - 1) / (float)g.fuse_max;
            danger = fmaxf(danger, wgt);
            break;                            // 泡挡火：看见它即止步
          }
        }
      }
    }

    // 0..P-1：各角色位置（双线性 splat）；P..2P-1：各角色名下泡泡的引信。
    // 泡泡按 owner 分通道而不是合并成一条"对手泡泡"：合并需要求和，
    // 求和一出现，视角就不再是纯置换，权重重排的技巧立刻失效。
    for (int p = 0; p < g.p; ++p) {
      base[p * ncell + cell] =
          (T)(al[p] ? splat_weight(py[p], px[p], row, col, g) : 0.f);
      base[(g.p + p) * ncell + cell] =
          (T)((o0 == p && f0 > 0) ? (float)f0 / (float)g.fuse_max : 0.f);
    }
    base[(2 * g.p + 0) * ncell + cell] = (T)(passable ? 0.f : 1.f);
    base[(2 * g.p + 1) * ncell + cell] = (T)danger;
    base[(2 * g.p + 2) * ncell + cell] = (T)progress;
  }
}

// ============ legal_mask：一线程一关卡，输出 (N,P,5) 方向 + (N,P,2) 放泡 ============
// 方向掩码标的是"按了也一格都动不了"，所以要真的跑一遍单轴碰撞消解。
// MOVE_IDLE 和 bomb=0 永远合法 ⇒ 不可能出现全掩码，格子版那个兜底分支没了。
__global__ void mask_kernel(const unsigned char* __restrict__ wall,
                            const int* __restrict__ fuse,
                            const signed char* __restrict__ owner,
                            const float* __restrict__ pos,
                            const unsigned char* __restrict__ alive,
                            unsigned char* __restrict__ mmask,
                            unsigned char* __restrict__ bmask, const Geom g) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= g.num_envs) return;
  const int ncell = g.h * g.w, N = g.num_envs;

  for (int p = 0; p < g.p; ++p) {
    unsigned char* m = mmask + ((size_t)env * g.p + p) * N_MOVES;
    unsigned char* b = bmask + ((size_t)env * g.p + p) * N_BOMB;
    b[0] = 1;
    if (!alive[idx2(p, env, N)]) {          // 死亡角色两个头都整行放开
      for (int a = 0; a < N_MOVES; ++a) m[a] = 1;
      b[1] = 1;
      continue;
    }
    const float y = pos[idx2(p * 2 + 0, env, N)];
    const float x = pos[idx2(p * 2 + 1, env, N)];
    for (int a = 0; a < 4; ++a) {
      float dy = (float)kDRow[a] * g.step_len;
      float dx = (float)kDCol[a] * g.step_len;
      float moved;
      if (dy != 0.f) {
        moved = fabsf(resolve_axis(y + dy, dy, x, y, x, wall, fuse, env, g, true) - y);
      } else {
        moved = fabsf(resolve_axis(x + dx, dx, y, y, x, wall, fuse, env, g, false) - x);
      }
      m[a] = (moved > 2.f * COLLIDE_EPS) ? 1 : 0;
    }
    m[MOVE_IDLE] = 1;
    int live = 0;
    for (int c = 0; c < ncell; ++c) {
      int i = idx2(c, env, N);
      if (owner[i] == p && fuse[i] > 0) ++live;
    }
    int cr = (int)floorf(y), cc = (int)floorf(x);
    int here = idx2(cr * g.w + cc, env, N);
    b[1] = (fuse[here] <= 0 && live < g.max_bombs) ? 1 : 0;
  }
}

// ================================ host 侧入口 ================================

static Geom make_geom(int h, int w, int p, int num_envs, int fuse_max, int blast,
                      int max_bombs, int max_steps, int max_chain, double step_penalty,
                      double radius, double step_len, double hit_reward,
                      double win_bonus, double danger_penalty,
                      double passivity_penalty, int passivity_ticks,
                      double max_hp, int win_hp_scaled) {
  TORCH_CHECK(p >= 2 && p <= MAX_P, "n_players 必须在 2..4");
  Geom g;
  g.h = h; g.w = w; g.p = p; g.num_envs = num_envs;
  g.fuse_max = fuse_max; g.blast = blast; g.max_bombs = max_bombs;
  g.max_steps = max_steps; g.max_chain = max_chain;
  g.step_penalty = (float)step_penalty;
  g.radius = (float)radius; g.step_len = (float)step_len;
  g.hit_reward = (float)hit_reward; g.win_bonus = (float)win_bonus;
  g.danger_penalty = (float)danger_penalty;
  g.passivity_penalty = (float)passivity_penalty;
  g.passivity_ticks = passivity_ticks;
  g.max_hp = (float)max_hp;
  g.win_hp_scaled = win_hp_scaled;
  return g;
}

void step_cuda(torch::Tensor wall, torch::Tensor fuse, torch::Tensor owner,
               torch::Tensor pos, torch::Tensor alive, torch::Tensor hp,
               torch::Tensor since_bomb, torch::Tensor t,
               torch::Tensor actions, torch::Tensor reward, torch::Tensor done,
               torch::Tensor covered, torch::Tensor trig, torch::Tensor expanded,
               int h, int w, int p, int num_envs, int fuse_max, int blast,
               int max_bombs, int max_steps, int max_chain, double step_penalty,
               double radius, double step_len, double hit_reward,
               double win_bonus, double danger_penalty, double passivity_penalty,
               int passivity_ticks, double max_hp, int win_hp_scaled) {
  auto g = make_geom(h, w, p, num_envs, fuse_max, blast, max_bombs, max_steps,
                     max_chain, step_penalty, radius, step_len, hit_reward,
                     win_bonus, danger_penalty, passivity_penalty,
                     passivity_ticks, max_hp, win_hp_scaled);
  const int threads = 256;
  const int blocks = (num_envs + threads - 1) / threads;
  step_kernel<<<blocks, threads>>>(
      wall.data_ptr<unsigned char>(), fuse.data_ptr<int>(),
      owner.data_ptr<signed char>(), pos.data_ptr<float>(),
      alive.data_ptr<unsigned char>(), hp.data_ptr<unsigned char>(),
      since_bomb.data_ptr<int>(), t.data_ptr<int>(), actions.data_ptr<int>(),
      reward.data_ptr<float>(), done.data_ptr<unsigned char>(),
      covered.data_ptr<unsigned char>(), trig.data_ptr<unsigned char>(),
      expanded.data_ptr<unsigned char>(), g);
  C10_CUDA_CHECK(cudaGetLastError());
}

void observe_cuda(torch::Tensor wall, torch::Tensor fuse, torch::Tensor owner,
                  torch::Tensor pos, torch::Tensor alive, torch::Tensor t,
                  torch::Tensor obs, int h, int w, int p, int num_envs,
                  int fuse_max, int blast, int max_steps, int n_ch) {
  auto g = make_geom(h, w, p, num_envs, fuse_max, blast, 1, max_steps, 1,
                     0.0, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0);
  const int threads = std::min(1024, ((h * w + 31) / 32) * 32);
  TORCH_CHECK(obs.is_cuda() && obs.is_contiguous(),
              "obs 必须是 contiguous CUDA tensor");
  TORCH_CHECK(obs.scalar_type() == torch::kHalf ||
              obs.scalar_type() == torch::kFloat,
              "obs dtype 只支持 fp16/fp32");
  TORCH_CHECK(n_ch == 2 * p + 3, "共享观测通道数必须是 2P+3");
  // obs 的 dtype 跟着 cfg.obs_fp16 走；两条分支只差写出时的 cast
  if (obs.scalar_type() == torch::kHalf) {
    observe_kernel<at::Half><<<num_envs, threads>>>(
        wall.data_ptr<unsigned char>(), fuse.data_ptr<int>(),
        owner.data_ptr<signed char>(), pos.data_ptr<float>(),
        alive.data_ptr<unsigned char>(), t.data_ptr<int>(),
        obs.data_ptr<at::Half>(), g, n_ch);
  } else {
    observe_kernel<float><<<num_envs, threads>>>(
        wall.data_ptr<unsigned char>(), fuse.data_ptr<int>(),
        owner.data_ptr<signed char>(), pos.data_ptr<float>(),
        alive.data_ptr<unsigned char>(), t.data_ptr<int>(),
        obs.data_ptr<float>(), g, n_ch);
  }
  C10_CUDA_CHECK(cudaGetLastError());
}

void mask_cuda(torch::Tensor wall, torch::Tensor fuse, torch::Tensor owner,
               torch::Tensor pos, torch::Tensor alive, torch::Tensor mmask,
               torch::Tensor bmask, int h, int w, int p, int num_envs,
               int max_bombs, double radius, double step_len) {
  auto g = make_geom(h, w, p, num_envs, 1, 1, max_bombs, 1, 1, 0.0, radius,
                     step_len, 0.0, 0.0, 0.0, 0.0, 0);
  const int threads = 256;
  const int blocks = (num_envs + threads - 1) / threads;
  mask_kernel<<<blocks, threads>>>(
      wall.data_ptr<unsigned char>(), fuse.data_ptr<int>(),
      owner.data_ptr<signed char>(), pos.data_ptr<float>(),
      alive.data_ptr<unsigned char>(), mmask.data_ptr<unsigned char>(),
      bmask.data_ptr<unsigned char>(), g);
  C10_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("step", &step_cuda, "批量推进一个 tick");
  m.def("observe", &observe_cuda, "写出 (N,2P+3,H,W) 共享观测");
  m.def("mask", &mask_cuda, "写出 (N,P,5) 方向掩码与 (N,P,2) 放泡掩码");
}





