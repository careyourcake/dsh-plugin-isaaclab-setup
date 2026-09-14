# Isaac Lab 环境搭建与运行（机器人仿真 + RL 训练）

目标：在 Linux + NVIDIA RTX 服务器上把 **Isaac Sim + Isaac Lab** 装对、跑通环境、训练 PPO/DreamerV3，并避坑无头渲染与远程运维。所有版本号和坑都来自真机踩坑，**先读「版本矩阵」再动手，否则必装坏**。

## 0. 版本矩阵（先看这里，别乱装）

| 组件 | 正确版本 | 错误版本 / 后果 |
|---|---|---|
| Isaac Sim | 5.0 / 5.1 | 老版本 API 不兼容 |
| Isaac Lab | 2.3.x（`isaaclab==0.47.2`） | — |
| 配套包 | `isaaclab_assets`、`isaaclab_tasks`、`isaaclab_rl==0.4.4` | 缺了 import 报 ModuleNotFoundError |
| PyTorch | 2.7.0（Isaac Sim 自带 pin 死） | 换版本会被 Isaac Sim 内部依赖覆盖 |
| **CUDA 架构** | sm_89（RTX 4090）✅ | **sm_120（RTX PRO 6000 Blackwell）❌ torch 2.7.0 不支持，会回退 CPU 或报错** |
| **rsl-rl** | `rsl-rl-lib==3.0.1` | `5.5.1` 改了 actor 配置格式，报 `KeyError: 'actor'`；`rsl-rl` 这个 PyPI 名不存在 |
| flatdict 编译 | `setuptools<81` + `pip --no-build-isolation` | 否则 `No module named 'pkg_resources'` |

**装之前先查 GPU 架构**：`nvidia-smi` 看显卡型号 → 对照 `torch.cuda.get_arch_list()`。sm_120（Blackwell）这一代是最大坑：Isaac Sim 5.1 钉死的 torch 2.7.0 只支持到 sm_90，装了 Blackwell 卡的机器上训练会静默跑 CPU。

## 1. 安装步骤（Linux）

```bash
# 1) 接受 EULA（两个都要，否则卡在 license 提示）
mkdir -p ~/.local/share/ov/pkg
touch ~/.local/share/ov/pkg/EULA_ACCEPTED
export OMNI_KIT_ACCEPT_EULA=Y    # 写进 ~/.bashrc，别只开在交互 shell

# 2) 官方 Installer 装 Isaac Sim（选带 Isaac Lab 的 bundle，或分开装）
# 3) 装 Isaac Lab（在独立 conda/venv，用 Isaac Sim 自带的 python）
pip install --no-build-isolation "setuptools<81"
pip install isaaclab==0.47.2 isaaclab_assets isaaclab_tasks isaaclab_rl==0.4.4
pip install rsl-rl-lib==3.0.1
```

验证：`python -c "import isaaclab; print(isaaclab.__version__)"`，再 `python -c "import isaaclab_rl; import isaaclab_tasks"`。

## 2. 代码骨架（必踩的坑）

### 2.1 AppLauncher 必须最先创建

`omni.client` / `carb` 只有在 `AppLauncher` 启动后才存在。**先建 app，再 import isaaclab.sim**：

```python
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)   # 自动带上 --headless 等
args = parser.parse_args()
app = AppLauncher(args)                      # 启动 Isaac Sim 内核
sim_app = app.app

import isaaclab.sim as sim_utils            # 必须在这之后
from isaaclab.assets import Articulation
```

### 2.2 DirectRLEnv 向量化环境

- prim path 用 **regex**：`UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")`。写 `{ENV_REGEX_NS}` 会直接报错（那是 ManagerBased 的占位符，DirectRLEnv 不认）。
- 场景配置：`InteractiveSceneCfg(num_envs=N, env_spacing=3.0, replicate_physics=True)`。
- `_reset_idx` / `write_root_pose_to_sim` 要**向量化**（batch 写，别传 `env_ids=[i]` 列表）。
- 自定义几何（墙/障碍）要在 `super().__init__()` 之前建好（`_setup_scene` 要用）。

### 2.3 headless 退出

`simulation_app.close()` 在 headless 下会**永久卡住**。脚本结尾用 `os._exit(0)` 硬退出。

### 2.4 无头渲染（截图/视频）

- headless kit（`isaaclab.python.headless.kit`）**没有 `omni.kit.viewport`**，`capture_viewport_to_file` 会 `ModuleNotFoundError`。
- 正确做法：用 `isaaclab.sensors.Camera` + `--enable_cameras`，`CameraCfg` 的 spawn 用 `PinholeCameraCfg`，位姿用 `camera.set_world_poses_from_view(eyes, targets)`；取图 `camera.data.output["rgb"]`（H,W,4）→ 丢掉 alpha 用 imageio 存 PNG。
- 纯几何俯视图可视化（走廊/扫掠体积）可以直接用 matplotlib，不必上 Isaac Sim。

## 3. 训练

### 3.1 PPO（rsl-rl）

```python
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg
# rsl-rl-lib==3.0.1，旧 policy 配置格式（不是 5.x 的 actor 格式）
```
- 环境包成 `RslRlVecEnvWrapper`，runner 写 policy 配置时用 `policy` 字段。

### 3.2 DreamerV3

- 用干净实现（如 NaturalDreamer）做骨干时，**奖励头必须用 symlog + two-hot**（DreamerV3 官方做法），别用朴素 Normal 分布——奖励量级跨越 2 个数量级（-100 擦墙 / +10 完成）时 Normal 头学不稳。
- symlog：`sign(x)*log1p(|x|)`；two-hot：255 桶 over `[-20,20]`，双桶加权交叉熵。
- 向量化环境交互：多环境并行收集，注意 isaaclab 的 `step()` 对终止环境返回的 obs 已经是 `_reset_idx` 之后的新观测（可直接当新回合起点）。

### 3.3 奖励设计先于一切

**先算清任务物理上可不可行**，再谈训练。例如四足机器人 0.65×0.37 做 90° L 转弯，理论最小走廊宽 ≈ 1.4m；环境默认参数给 1.0m 宽的话，任何算法都学不会（必擦墙）。训练前用纯几何（`signed_clearance`）验证最小间隙。

## 4. 远程服务器运维

- 后台训练：`setsid nohup python train.py --headless < /dev/null > train.log 2>&1 &`，`disown` 分离。
- **杀进程别用 `pkill -f train.py`**——`-f` 匹配整条命令行，会连你正在跑的 `bash -c "ssh ... pkill -f train.py ..."` 一起杀掉。用精确 pattern：`pkill -f 'python -u train.py'`，或 `ps aux | grep` 拿到 PID 再 `kill -9`。
- 远程执行（paramiko）：SSH 通道会在 `setsid ... &` 后台进程持有 fd 时保持不关，表现为命令"超时"——进程其实已经起了，重新连一次确认即可，别误判成失败。

## 5. 排错速查

| 症状 | 原因 / 解决 |
|---|---|
| `ModuleNotFoundError: omni.client` / `carb` | AppLauncher 没在 import 之前建 |
| `KeyError: 'actor'`（rsl-rl） | 装了 5.x，降级 `rsl-rl-lib==3.0.1` |
| `No module named 'pkg_resources'` | `setuptools<81` + `--no-build-isolation` |
| EULA 卡住 | 写 EULA_ACCEPTED 文件 + 环境变量 |
| `simulation_app.close()` 卡死 | 换 `os._exit(0)` |
| torch 警告 `sm_120 not compatible` | 显卡是 Blackwell，换 sm_89 的卡或降级需求 |
| `omni.kit.viewport` 找不到 | headless kit 无 viewport，用 Camera 传感器 + `--enable_cameras` |
| 训练 score 永远卡在最差值 | 先查任务几何是否物理不可行（见 3.3） |

## 6. 一句话流程

`查 GPU 架构 → 装 Isaac Sim 5.x + isaaclab 0.47.2 + rsl-rl-lib 3.0.1 → AppLauncher 先行 → DirectRLEnv(regex prim path) → 几何可行性先行 → 训练 → 无头渲染用 Camera`。
