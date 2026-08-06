# 预生成关卡（96 关）

骨架：corridor（顶 4 行永久墙、左右 5 列可炸墙、中间空旷区）。
空旷区随机可炸墙 + 少量永久墙柱（**每柱 BFS 校验，无死区**）。
用法：加载 level_XXXX.pt 的 wall/brick 替换 make_walls/make_bricks。
