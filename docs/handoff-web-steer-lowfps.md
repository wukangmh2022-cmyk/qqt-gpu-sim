# Codex 会话交接：网页端寻路/推箱修复 + low 帧定位（2026-08-27 ~ 08-28）

> 本文档由 ZCode 从 Codex 会话 `01a041e6-6528-7782-b3d5-bbd2efcd3249` 提取整理，
> 用于在 ZCode 侧续接开发。完整对话记录见
> `docs/sessions/2026-08-27-codex-01a041e6-transcript.txt`（164 条消息）。
> Codex 原始会话文件：`~/.codex/sessions/2026/08/27/rollout-2026-08-27T14-26-53-01a041e6-….jsonl`

## 任务背景

用户把网页端（`web/`）当作 JAX 训练模拟器（`jax_bomb/jax_env.py`）的**人类验收操控器**，
核心要求：网页端 `_steer()` 与 JAX 训练端行为完全一致——人类在网页上验证通过的动作语义，
就是 AI 训练时的动作空间语义（上下左右 + 停留，动作空间不变但语义升级为"目标相邻格 + 自动侧滑/推箱"）。

## 一、已完成（全部代码已随 commit `9e17eef` 入库，当前工作树干净）

### 1. `_steer()` 修复（web/sim.js 与 jax_bomb/jax_env.py 同步）
- 正前受阻、左右都可走时：按角色相对格中心偏移向中心线归中，不再固定优先左/上。
- pushable（可推箱）目标：禁止垂直侧滑，保持原方向累计 `push_t`（0.3s 推动阈值）；贴箱前允许直线贴近。
- JAX 侧：`legal_mask()` 不屏蔽 pushable 方向（PPO 可学）；`_steer(..., pushable=None)`；未新增动作。
- "部分直走"（碰撞盒擦到旁边砖只走半步）不再算成功，触发垂直修正；一格宽通道中心线钳制防左右振荡。
- 爆炸糖浆视觉时长 0.6s → 0.4s（只改视觉，伤害时序不变）。

### 2. 网页鼠标寻路（web/main.js）
- 九宫格 3×3 高亮（中心=停留），角格动态映射：对 4 个 Destination 用真实 `_steer()` 试算，
  选执行后离目标格中心最近者；无改善则不动（防"上一步/下一步"循环）。
- 后升级为最多 4 tick 短视野试算（允许"先对齐再进入"，解决贴大碰撞体时单步贪心误判 IDLE）。
- `mousemove` 只做轻量高亮判断，完整试算只在 `click`（修掉 4^4=256 次试算放 mousemove 的性能事故）。
- 侧边栏"鼠标寻路调试"开关（`#mouse-path`），**默认关闭**。
- 点击相邻可推箱自动持续方向 ~0.42s；键盘/摇杆输入立即取消。
- 角色朝向规则：实际移动与点击意图一致才更新朝向；侧滑/未动保持原朝向。

### 3. 选图页 UI（web/index.html + web/style.css）
- 选图与进入分离：点地图只选中，底部闪烁"点击进入"按钮才开局。
- 六个滑块：初始/最大 × 泡泡、威力、速度（作用于双方玩家，`sim.reset()` 后覆盖；初始不超过最大）。
- 滑块文字 11px（高优先级选择器覆盖 `#banner` 的 24px 继承）；删除重复属性小字。

### 4. 动作语义确认（结论：无需改）
- 动作持续到被下一动作替换；网页 `inferEvery>1` 缓存复用；JAX PPO 每 tick 采样为标准做法。

### 5. 性能 profiling（web/main.js）
- tick 分段计时：动作/快照/step/事件/dangerMap/post；`tickTimeline` 记录最近 200 次 tick 起止。
- 突变帧（>25ms）记录与哪些 tick 时间重叠及重叠毫秒数；Long Task/LoAF 只报与当前突变帧重叠者。
- 突变存 `window.__qqtProfSpikes`（最多 200 条）+ `canvas.dataset.profSpikes/profLastDt`。
- `?profquiet=1` 静默记录不打印；`?profconsole=1` 显式开每秒摘要。25ms 阈值实时告警保留。

### 6. low 帧问题结论（真实 Chrome A/B 实测）
- **已排除**：sim.step（0.1–0.3ms）、渲染（0.2ms）、dangerMap、鼠标寻路、音频/BGM、
  录像采样、GPU ONNX 推理（实际 2–3ms；日志里的 100–150ms 是 CPU 对照统计，勿误判）。
- **找到并修复一个真实来源**：`replay.frames.push(sim.snapshotReplay(info))` 在"录制动图"未勾选时
  也每 100ms 无条件复制二十多个地图数组永久堆积 → GC 周期性停顿。
  已改为仅勾选录像时采集，关闭即释放（A/B：静止敌人 15s 内 9 次 → 30s 1 次）。
- **主因**：固定 10Hz `setInterval` logicTick 与浏览器 vsync 相位撞车。
  `requestIdleCallback` 调度已设为默认（后台页仍走原逻辑）：
  Hunter 22→8 次/30s、模型 45→19 次/30s。**缓解但未清零**。
- Chrome 环境本身有 vsync 丢失基线：空白静默 rAF 探针页 30s 也偶发 5 次 >25ms
  （`web/frame_probe.html`，诊断用，任务结束可决定去留）。

## 二、已撤回的错误尝试（不要重蹈）

| 尝试 | 结果 |
|---|---|
| tick-vsync：tick 搬到 rAF 后执行 | 规律性 33.3ms，严重恶化，已完整回滚 |
| 60FPS 时间闸门 | 错误跳帧变 30fps，已回滚 |
| 关闭 profiler 默认输出 / 250ms 限频 | 用户明确否决，已恢复 25ms 实时告警 |
| 归因 DevTools 重绘/"打印器" | 错误归因（鸡生蛋问题），已承认 |
| 归因 GPU 推理/AI 计算量突变 | 用户早已指出静止/规则敌人也复现，属误判 |

## 三、未完成（ZCode 续接点）

用户在 Codex 的最后一条消息（2026-08-28 07:49）指出下面这个承诺没做完：

> "我会用自动开局参数重新跑一局，并直接读每个突变帧携带的 tickParts，
> 确认是 danger/step/事件 还是完全在页面 JS 之外。这个采样会给出可复核的原始数值。"

诊断代码已具备（`tickTimeline` + 突变帧 `overlappingTicks`，web/main.js:2267 附近），但采样验证没跑。待办：

1. 用自动开局参数（URL 带 `?auto=1`）重跑一局，读取每个突变帧的重叠 tick 与 tickParts，
   输出可复核数值表：突变到底与 tick 执行重叠，还是完全在页面 JS 之外（系统/合成器）。
2. idle 调度成为默认后剩余突变（模型 ~19、Hunter ~8 次/30s）是否还有页面内可控来源。
3. `web/frame_probe.html` 诊断页去留。

## 四、约束与沟通要求（用户非常明确）

- **必须信 prof 日志（frameDt>25ms），不要信右上角平均 FPS**（fps 是假象，low 帧手感是真实的）。
- 不要再往 GPU 推理、AI/tick 计算量方向归因（计算量恒定；"突变"是相位问题不是量的问题）。
- 保留 25ms 突变阈值实时告警，**不调阈值、不限频、不静音**。
- 改任何调度默认前，必须在用户真实 Chrome 里 A/B（`?profquiet=1` 计数对比），拿真实数据说话；
  没改善立即撤回，不叠加实验性改动。
- 不说"彻底修好/只剩一次"，除非长期 A/B 数据支持；承认不确定性。
- 用户 Chrome 已装 GPT 插件可被接管连接（当时通过 `agent.browsers.get("chrome")`）。
- 旧约束"工作树很脏不能 reset"已解除：改动已全部提交（`9e17eef`），当前树干净。

## 五、验证命令

```bash
node --check web/main.js
node --check web/sim.js
node web/test_mouse_steer.js
node scripts/test_replay_roundtrip.js
python3 -m py_compile jax_bomb/jax_env.py tests/test_jax_steer.py
git diff --check
```

本机无 pytest/JAX，`tests/test_jax_steer.py` 只能做 py_compile（逻辑与 JS 版对齐）。

## 六、相关文件

- `web/main.js` — 鼠标寻路、profiling、调度（当前版本 `main.js?v=20260828-idletick-default`）
- `web/sim.js` — `_steer()`、推箱、爆炸视觉（当前版本 `sim.js?v=20260827-steer-prof`）
- `web/index.html` / `web/style.css` — 开关、选图 UI、滑块
- `jax_bomb/jax_env.py` — JAX 侧 `_steer`/`legal_mask` 同步修复
- `web/test_mouse_steer.js` / `tests/test_jax_steer.py` — 回归测试
- `web/frame_probe.html` — 空白 rAF 探针（诊断用）
