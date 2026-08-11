# MiniMax H3 · 参考图→视频 (ref2va) 链路说明

> ComfyUI (Comfy-Org/MiniMax-H3 @comfy 0.30) 参考图视频链路，跑在实验室 DTK DCU 上，与 course_bc 训练共存（--lowvram 弹出加载，训练占用可忽略）。

## 链路拓扑

```
参考图(1~9张) ──► LoadImage ──┐
                              ├──► MiniMaxH3ReferenceToVideo ─(positive)→ BasicGuider
text: qwen3vl encoder(int8) ──┼──► clip                                   │
video VAE / audio VAE ────────┴──► vae / audio_vae ────────┘               ├─► Sampler(res_multistep)+BasicScheduler(beta)
                                                                           │      │ model = ref2va(int8, 20.97G)
RandomNoise ──────────────────► noise ────────────────────────────────────┘
      ▼
SamplerCustomAdvanced ──► VAEDecode(video) ──┐
                      └─► VAEDecodeAudio ─────┼──► CreateVideo(24fps) ─► SaveVideo → output/video/h3_r2v_*.mp4
```

要点（与文生视频 fl2v 的差别）：
- 扩散模型权重换为 **`minimax_h3_ref2va_pruned_int8_convrot.safetensors`**（20.97G，与 fl2v int8 同架构、同大小，独立权重集）
- 节点为 `MiniMaxH3ReferenceToVideo`，**必须**给 audio_vae；ref 按顺序 `<Picture 1>` 用 tag 引用
- 参考 tokens 全程参与每步采样，**比文生视频明显更慢**；`ref_image_size=m Closed`比 `match` 更稳但慢数倍
- 采样器推荐 `res_multistep` + scheduler `beta`（参考提示词场景）

## 搭建（一次性）

```bash
# 1) 拉权重（6路分片流式，本地零落盘）→ 服务器 /root/minimax_h3/
bash /tmp/h3_pipe_r2v.sh          # ETA 与带宽相关，完成后字节级校验 wc -c

# 2) 服务器放行给 ComfyUI 认识（建符号链接）
sshpass -p 'YOUR_PASSWORD（DCU_PASS，见根目录 .env）' ssh -p 10630 root@ssh.zzai.scnet.cn \
  "ln -sf /root/minimax_h3/minimax_h3_ref2va_pruned_int8_convrot.safetensors \
          /root/comfy/models/diffusion_models/ && ls -la /root/comfy/models/diffusion_models/"

# 3) 重启 ComfyUI（folder_paths 启动时缓存模型列表，必须重启才能看到新文件）
bash /tmp/h3_comfy_restart.sh
```

## 日常一条命令

```bash
cd qqt-gpu-sim/scripts
./h3_r2v_run.sh /path/to/ref.png \
  --prompt "<Picture 1> 保持参考图的角色与配色，镜头缓缓推近，背景加入夜色霓虹，人物小幅喘气动作，氛围环境声＋低音配乐，无字幕无水印。" \
  --size 832x480 --sec 5 --steps 20 --seed 20260808
# 完成后会在当前目录留下 h3_r2v_<ts>.mp4
```

参数：
| 参数 | 默认 | 说明 |
|---|---|---|
| `--sec` | 5 | 时长秒，自动对齐模型 17k+5 栅格 |
| `--size` | 832x480 | 16:9；短边 ≤768，多用 832x480 / 1280x736 / 1344x768 |
| `--steps` | 20 | 步数，参考类建议 20-30 |
| `--seed` | 时间戳 | 随机种子 |
| `--ref-size` | match | `match`快 / `max`(≤2048 短边)更保真但慢数倍 |
| `--scheduler` | beta | `beta`/`normal` 供参考场景 |

## 引用 tag 语法

参考输入按连接顺序编号（ref_images → ref_videos → ref_audios），提示词里写：

- `<Picture 1>`、`<Picture 2>` … 对应上传的第 1、2 张参考图
- `<Video 1>` 参考视频；`<Audio 1>` 参考音频（该轨道映射范式）
- 描述 目标场景/动作/音频时要明确是“哪张图”陪哪段

## 其他验证方式（服务器直连）

```bash
# 上传图片（WebUI upload 方式，返回文件名）
curl -F "image=@/path/ref.png" http://127.0.0.1:8188/upload/image
# 查 job
curl -s http://127.0.0.1:8188/history/<prompt_id>
# 出片在 /root/comfy/output/video/
```

## 与训练共存

- ComfyUI 以 `--lowvram` 运行，只有当前采样块占用 GPU，训练（定基本 6-8G）可并行
- 视频生成更重点（ref tokens 全程采样），若与训练强抢占 GPU，可临时降低 `--sec`/`--size`/`steps`

## 已验证

- ref2v int8 权重字节校验 ✓；ComfyUI 0.30 原生节点 `MiniMaxH3ReferenceToVideo` ✓
- 全链路首支 H3 视频（文生视频 fl2v）已出（h3_first_test.mp4）