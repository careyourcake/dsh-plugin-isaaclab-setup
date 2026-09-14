# dsh-plugin-isaaclab-setup

DeepSeek Harness plugin：在 Linux + NVIDIA 服务器上搭建并运行 **Isaac Sim + Isaac Lab**，训练机器人 RL（PPO / DreamerV3）。

打包一个 skill：**`isaaclab-setup`**，把真机踩坑沉淀成可复用知识——版本钉死、AppLauncher 先行、DirectRLEnv 向量化、无头渲染、远程训练运维。

## 安装

```sh
dsh plugin add dsh-plugin-isaaclab-setup
```

安装后 `isaaclab-setup` 技能进入技能目录，agent 在「装 Isaac Lab / 写机器人 RL 环境 / 训练 PPO·DreamerV3 / 修 Isaac Sim 报错 / 无头渲染 / 远程训练」时会自动调用。

## Skill 覆盖内容

| 章节 | 内容 |
|---|---|
| 0. 版本矩阵 | `isaaclab==0.47.2`、`isaaclab_rl==0.4.4`、`rsl-rl-lib==3.0.1`、CUDA sm_89 ✅ / sm_120 ❌、EULA |
| 1. 安装步骤 | EULA 接受、setuptools<81、pip 顺序 |
| 2. 代码骨架 | AppLauncher 先行、DirectRLEnv regex prim path、向量化 reset、headless 退出、无头渲染用 Camera |
| 3. 训练 | PPO(rsl-rl) 配置、DreamerV3 symlog+two-hot 奖励头、几何可行性先行 |
| 4. 远程运维 | setsid nohup 分离、pkill -f 自匹配陷阱、paramiko 超时误判 |
| 5. 排错速查 | 8 个常见症状 → 原因 → 解决 |

## 目录

```
skills/isaaclab-setup.md   # skill 正文
lib/index.js               # Cordis skill provider（注册到 ctx.skills）
cordis.patch.yml           # bundle patch（dsh plugin add 自动激活）
```

## License

MIT
