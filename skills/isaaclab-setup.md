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

### 2.4 无头渲染（3D 截图/视频）

- headless kit（`isaaclab.python.headless.kit`）**没有 `omni.kit.viewport`**，`capture_viewport_to_file` 会 `ModuleNotFoundError`。
- **Camera 必须放进 `InteractiveScene`**（`CameraCfg` + `scene["camera"]`），由 `scene.reset()` 自动初始化——不要 standalone `Camera(...)` 后手动调 `_initialize_impl()`（会挂）。
- 启动加 `--enable_cameras`；`CameraCfg` 的 spawn 用 `PinholeCameraCfg`；位姿用 `scene["camera"].set_world_poses_from_view(eyes, targets)` 对准目标；取图 `scene["camera"].data.output["rgb"][0,...,:3]`（H,W,3）→ imageio 存 PNG / `mimsave` 存 GIF。
- **灯光坑（大）**：`DomeLightCfg(intensity=1.0)` 是默认亮度、画面正常；**往大了设（2000/6000/10000）画面反而全黑**（这个版本的物理单位/tone-mapping 与直觉相反）。别抄老教程里的 `intensity=2000`。
- 机器人摆位：`scene["robot"].write_root_pose_to_sim(pose)` + `write_joint_state_to_sim(...)` + `scene.write_data_to_sim()` + `sim.step()` + `scene.update(dt)`，再 `camera.update(dt=0)` 取帧。
- 纯几何俯视图（走廊/扫掠体积/地形 heightmap）直接用 matplotlib，不必上 Isaac Sim。

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

## 4. 环境生成 recipe（地形 + 观测/奖励模板 + 验证清单）

生成一个高质量 RL 环境的固定流程：**地形 → 机器人 → 观测/奖励 → 验证**。地形生成器都在 `isaaclab.terrains` 下，分两类：

- **height_field**（`isaaclab.terrains.height_field`）：先生成 2D 高度场数组，再转 mesh，适合粗糙地形。
- **trimesh**（`isaaclab.terrains.trimesh`）：直接生成 mesh，适合结构化障碍（台阶/箱/坑/轨道/间隙）。

### 4.1 地形生成 + headless 可视化的正确姿势

坑：`trimesh.Scene.save_image()` 会调用 pyglet 的 `SceneViewer`，**headless 服务器上直接 `NoSuchDisplayException`**。正确做法是拿**原始 2D 高度场**用 matplotlib 画：

```python
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app            # 先起 app，再 import isaaclab

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from isaaclab.terrains.height_field import hf_terrains, hf_terrains_cfg

cfg = hf_terrains_cfg.HfPyramidStairsTerrainCfg(
    size=(10.0, 10.0), horizontal_scale=0.1, vertical_scale=0.005,
    border_width=0.5, step_height_range=(0.05, 0.15), step_width=0.3, platform_width=1.5)
# cfg.function 被 @height_field_to_mesh 装饰过，返回 (meshes, origin)
# 用 .__wrapped__ 拿原始函数 → 2D 离散高度场；× vertical_scale 才是米
hf = hf_terrains.pyramid_stairs_terrain.__wrapped__(difficulty=1.0, cfg=cfg) * cfg.vertical_scale
plt.imshow(hf.T, origin="lower", cmap="terrain"); plt.colorbar(label="height (m)"); plt.savefig("stairs.png")
```

> **关键坑：heightfield 数组 ≠ 物理碰撞体**。height_field 函数返回的 2D 数组只是**中间产物**——`@height_field_to_mesh` 装饰器会调 `convert_height_field_to_mesh()` 把它转成 `trimesh.Trimesh`（三角网格），**物理引擎里机器人踩的是这个 mesh，不是 heightfield 数组**。转换时用 `slope_threshold` 把超过阈值的陡峭竖直面「掰成斜边」，消除相邻网格高度不连续的「接缝」，否则四足脚尖会卡进网格接缝里。所以送进物理前要检查的是 **mesh（三角面连续、无长竖直面）**，不是数组。

> **mesh 地形（gap / box / pit / rails / 间隙/箱/坑/轨道）没有高度场数组**：`mesh_terrains.gap_terrain(difficulty, cfg)` 直接返回 `(meshes, origin)`（trimesh 列表）。headless 可视化用**解析式高度图**（按几何把「间隙」格点置 NaN，见 `MeshGapTerrainCfg`：需 `gap_width_range` + `platform_width`），或用 trimesh ray casting 光栅化。`.__wrapped__` 只对 height_field 有效。

### 4.2 可生成地形清单

| 类别 | 类名 | 说明 |
|---|---|---|
| height_field | `HfRandomUniformTerrainCfg` | 随机起伏 |
| height_field | `HfPyramidSlopedTerrainCfg` / `HfInvertedPyramidSlopedTerrainCfg` | 金字塔斜坡（上/下） |
| height_field | `HfPyramidStairsTerrainCfg` / `HfInvertedPyramidStairsTerrainCfg` | 楼梯（上/下） |
| height_field | `HfDiscreteObstaclesTerrainCfg` | 离散障碍 |
| height_field | `HfWaveTerrainCfg` | 波浪 |
| height_field | `HfSteppingStonesTerrainCfg` | 踏脚石（梅花桩） |
| mesh | `MeshPlaneTerrainCfg` | 平面 |
| mesh | `MeshPyramidStairsTerrainCfg` / `MeshInvertedPyramidStairsTerrainCfg` | 楼梯（上/下） |
| mesh | `MeshRandomGridTerrainCfg` | 随机网格 |
| mesh | `MeshRailsTerrainCfg` | 轨道 |
| mesh | `MeshPitTerrainCfg` | 坑 |
| mesh | `MeshBoxTerrainCfg` | 箱子 |
| mesh | `MeshGapTerrainCfg` | 间隙 |
| mesh | `MeshFloatingRingTerrainCfg` | 浮环 |
| mesh | `MeshStarTerrainCfg` | 星形 |
| mesh | `MeshRepeatedPyramids/Boxes/CylindersTerrainCfg` | 重复金字塔/箱/圆柱 |
| 自定义 | 手动 `sim_utils.CuboidCfg` / `GroundPlaneCfg` | 走廊、任意墙段（如 MIRAGE 的 L 拐角） |

### 4.3 完整环境模板（复制即用）

**不要从零手写 env**——直接用模板 `templates/direct_rl_env.py`（Go2 + 可配置地形 + 完整观测/奖励/终止，可直接训）。改 `TERRAIN_ZOO` 里选哪种地形、改 `_get_rewards` 加奖励项即可。

模板里的观测/奖励标准做法（四足速度跟踪任务）：

| 项 | 内容 |
|---|---|
| 观测 (34维) | base 线速度(3) + 角速度(3) + 投影重力(3) + 关节位置偏差(12) + 关节速度(12) + 速度指令(1) |
| 奖励 | 速度跟踪 `exp(-(vx-cmd)²/0.25)` ×1.5；罚：角速度²、关节速度²、动作率²、非竖直姿态、base 接触 |
| 终止 | base 接触(摔倒) / 超时 |
| 重置 | 随机速度指令 + 默认站姿 |

**奖励量级铁律**：跟踪项 ±1~2、罚项 -0.01~-1。**避免像「擦墙 -100」那种跨 2 个数量级的悬崖**（会毁掉 DreamerV3，见 3.2/3.3）。

关键 API（模板已用）：
- `self.robot.data.root_lin_vel_b / root_ang_vel_b / projected_gravity_b / joint_pos / joint_vel`
- `self.robot.set_joint_position_target(...)`（PD 保持）
- `self.contact_sensor.data.net_forces_w_history[:,-1]...`（接触判定）

### 4.4 3D 渲染 + 可视化（截图/视频/GIF）

**用模板 `templates/render_scene.py`**（地形 + Go2 + 第三视角 RGB + 右下角 Depth Image 内嵌 + 步态动画 → GIF）。要点：

1. **Camera 走 `InteractiveScene`**（`CameraCfg` + `scene["camera"]`），加 `--enable_cameras`。
2. **灯光**：`DomeLightCfg(intensity=1.0)`（正常）+ `DistantLightCfg(intensity=8, rot=(0.87,0,0.5,0))`（主光源，出方向阴影）。
3. **深坑/间隙**：地形下方加一块暗色地板（z=-2），否则 gap 透出背景显白。
4. **深度相机**：`data_types=["distance_to_image_plane"]`，输出 `(H,W,1)` → **`reshape(H,W)`** 才能 `Image.fromarray`。
5. **内嵌小图**：深度归一化（近白远黑）→ resize → `PIL` 贴到右下角 + 白底 + "Depth Image" 标题。
6. **机器人摆位**：`write_root_pose_to_sim` + `set_joint_position_target(default)` + `scene.write_data_to_sim()` + `sim.step()` + `scene.update(dt)`。
   - **绝对不要写死 base 的 z**：不同地形平台高度不同（楼梯平台高、gap 平台≈0）。先高抛到 z=2.0，PD 保持站姿跑 ~400 步让它自然落到地形上，**读实际 `root_pos_w[0,2]` 当站立高度**，相机也跟着这个高度来定，否则机器狗会埋进地形或跑出画面。
7. **摆位索引坑**：`default_joint_pos` 是 `(num_envs, num_joints)`，改关节要写 `target[0, idx]`，不是 `target[idx]`。
8. **步态动画**：对角小跑（FL/RR 同相、FR/RL 反相），大腿 `default + 0.5*sin(2πft+phase)`、小腿 `default - 0.35*max(0,cos(...))`，base 沿 +x 平移。
9. 多机位一次渲染对比（`cams` 列表循环），挑最好的角度，别一次只试一个。
10. **输出 MP4**：服务器一般有 ffmpeg，`imageio.mimsave("out.mp4", frames, fps=25)` 自动走 ffmpeg 后端；GIF 体积大且掉帧，优先 MP4。
11. **地形特征（台阶/坡）看不见怎么办**——两个反直觉要点：
    - **相机别太高**：俯角太陡只看到踏面，台阶看着像平地；要用**低角度**看台阶**立面**（eye 只比目标高 0.15~0.5m）。
    - **环境光别太强**：`DomeLight` 会把台阶阴影冲平；用**弱环境光(0.25~0.75) + 强平行光(8~14) + 大台阶(step_height 0.14~0.22)**，台阶立面才有明暗对比。

### 4.5 验证清单（生成完必做）

1. 地形：heightfield 先 matplotlib 可视化，确认几何对、z 范围合理（mesh 地形看解析式高度图）。
2. 机器人：spawn 后 `step()` 几帧，**obs 无 NaN、reward 在预期量级、termination 按预期触发**（`torch.isnan(obs).any()`）。
3. 渲染：渲一帧确认机器人可见、相机框正、灯光正常（别全黑）。
4. 可行性：任务几何物理可解（3.3）。

### 4.6 工作流：需求 → 环境（照这个走）

> 例：用户说"生成一个楼梯环境"。

1. **选地形**：查 4.2 清单 → 楼梯选 `MeshPyramidStairsTerrainCfg`（mesh，物理用三角网格不会卡缝）或 `HfPyramidStairsTerrainCfg`。
2. **配参数**：步高 `step_height_range`、步宽 `step_width`、平台 `platform_width`。**注意 `MeshGapTerrainCfg` 等 mesh 地形不接受 `border_width`**（只有部分接受）。
3. **验证几何**：先 4.1 出 heightmap 图，确认台阶数/高度合理。
4. **建环境**：复制 `templates/direct_rl_env.py`，`TERRAIN_ZOO` 换成楼梯配置。
5. **跑通验证**：`python direct_rl_env.py --terrain stairs --headless --num-envs 4`，看 obs/reward 无 NaN、termination 正常。
6. **渲染确认**：复制 `templates/render_scene.py`，`ZOO` 换楼梯，渲染 GIF 确认机器人站/走在楼梯上。
7. 训练：包 `RslRlVecEnvWrapper` 接 PPO（见 3.1）。

## 5. 远程服务器运维

- 后台训练：`setsid nohup python train.py --headless < /dev/null > train.log 2>&1 &`，`disown` 分离。
- **杀进程别用 `pkill -f train.py`**——`-f` 匹配整条命令行，会连你正在跑的 `bash -c "ssh ... pkill -f train.py ..."` 一起杀掉。用精确 pattern：`pkill -f 'python -u train.py'`，或 `ps aux | grep` 拿到 PID 再 `kill -9`。
- 远程执行（paramiko）：SSH 通道会在 `setsid ... &` 后台进程持有 fd 时保持不关，表现为命令"超时"——进程其实已经起了，重新连一次确认即可，别误判成失败。

## 6. 排错速查

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
| `trimesh.Scene.save_image` 报 `NoSuchDisplayException` | headless 无 X 显示；改用 `.func.__wrapped__` 拿高度场 + matplotlib（见 4.1） |
| 渲染全黑（mean<40） | `DomeLightCfg(intensity=...)` 设太大（2000+）；改回 **1.0**（见 2.4） |
| `MeshXxxTerrainCfg.__init__() got an unexpected keyword argument 'border_width'` | 部分 mesh 地形（如 `MeshGapTerrainCfg`）没有 `border_width` 字段，删掉该参数；border 由 `TerrainGeneratorCfg.border_width` 控制 |
| 深度图合成报 `Cannot handle this data type: (1,1,1,3)` | 深度输出是 `(H,W,1)`，先 `reshape(H,W)` 再 `Image.fromarray`（见 4.4） |
| 改关节 `target[idx]` 报 index out of bounds | `default_joint_pos` 是 `(num_envs, num_joints)`，写 `target[0, idx]`（见 4.4） |
| 机器人一 spawn 就瘫倒 | 没做 PD 保持；每步 `set_joint_position_target(default_joint_pos)`（见 4.3/4.4） |
| standalone `Camera` 挂住 | Camera 必须进 `InteractiveScene`（见 2.4） |
| DirectRLEnv 里 prim path 报 `is not global` | 用了 `{ENV_REGEX_NS}`（ManagerBased 占位符）；DirectRLEnv 要用 `/World/envs/env_.*/Robot`（见 2.2/4.3） |
| 渲染时机器狗埋进地形/跑出画面 | base z 写死了；先 settle 读实际站高，相机也按站高定（见 4.4） |

## 7. 一句话流程

`查 GPU 架构 → 装 Isaac Sim 5.x + isaaclab 0.47.2 + rsl-rl-lib 3.0.1 → AppLauncher 先行 → 环境生成(需求→环境走 4.6 七步：选地形 4.2 → 配参 → heightmap 验证 4.1 → 复制 env 模板 4.3 → 跑通验证 4.5 → 渲染确认 4.4) → 训练 3 → 运维 5`。

**两个模板**（`templates/` 下直接复制改）：
- `direct_rl_env.py`：完整 DirectRLEnv（地形+Go2+观测/奖励/终止，可训）。
- `render_scene.py`：3D 渲染（地形+机器人+深度内嵌+步态 GIF）。
