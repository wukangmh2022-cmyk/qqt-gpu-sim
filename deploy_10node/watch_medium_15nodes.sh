#!/usr/bin/env bash
# 实时监控 15 节点 30 卡长训状态
set -euo pipefail

sshpass -p '0HLD3RHKXNBI5X5' ssh -o StrictHostKeyChecking=no -p 10572 root@ssh.zzai.scnet.cn "tail -n 25 /root/private_data/train_r0.log"
