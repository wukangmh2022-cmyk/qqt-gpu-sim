# Legacy 归档目录说明

本目录归档了项目早期（8月中上旬）探索、试错与不同硬件平台（DCU、昇腾 910B、Torch、Triton 等）的遗留脚本，供后续复盘与技术参考，避免污染根目录与主训练流。

## 目录结构

### 1. `probes_and_profilers/`
- **内容**：早期针对 DCU 图模式、昇腾 910B、Triton 算子、JAX CNN 算子等隔离测试与性能探针。
- **包含**：`iso_*.py`、`probe_*.py`、`prof_*.py`、`verify_triton_*.py`、`verify_inductor_dcu.py`、`verify_legacy_dcu.py` 等。

### 2. `sbatch_and_cluster/`
- **内容**：早期单机、多卡与 Slurm 集群调度历史脚本（如早期 JAX sbatch 批次、课程学习启动脚本等），已被 `scripts/launch_8node_*.sh` 统一标准脚本替代。
- **包含**：`run_jax*.sbatch`、`run_ppo*.sbatch`、`run_train_*.sh`、`start_*.sh`、`dcu_*.sh` 等。

### 3. `models_and_eval/`
- **内容**：早期针对 LSTM 模型、纯 CNN 模型、单卡 CPU 的简单对打与评测脚本，已被 `scripts/eval_headless_parallel.js` 统一标准套件替代。
- **包含**：`eval_cnn_bots.py`、`eval_lstm_*.py`、`eval_npu.py`、`train_lstm_speed.py`、`test_raw_vs_ema.py`、`headless_test_old.js` 等。
