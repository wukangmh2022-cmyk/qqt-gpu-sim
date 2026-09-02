# 训练实验配置

长训使用带注释的 TOML 文件作为参数清单，例如：

```bash
python3 -m jax_bomb.train_real --config configs/repro_it68_asym_timeout_k64.toml
```

`scripts/launch_8node_it68_hlgauss_top25_patch3.sh` 默认读取同一份配置；
平台只负责注入 `WORLD_SIZE`、`RANK`、`MASTER_ADDR` 和 `MASTER_PORT`。

配置优先级为：命令行参数 > 配置文件 > 环境变量 > 程序默认值（启动脚本
中的 checkpoint 路径环境变量也是如此）。checkpoint 的 `cfg` 会记录最终
解析后的数值以及配置文件绝对路径，便于确认实验没有漏参数。

修改实验前重点检查三组字段：

- `[run]` 的 `iters`：最终 iteration / 产物终点。
- `[reward]` 的终局奖励、超时奖励和 `*_anneal_*`：决定奖励时间表。
- `[rollout]` 与 `[distributed]`：决定每卡负载、总 SPS、Local-SGD 的 K
  和通信量；改卡数时同时核算全局 `num_envs` 与 `minibatch`。
