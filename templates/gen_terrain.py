#!/usr/bin/env python3
"""生成并可视化任意 height_field 地形（headless 安全）。

用法（在 Isaac Lab 服务器上）：
  python gen_terrain.py --terrain pyramid_stairs --headless
  python gen_terrain.py --terrain wave
  python gen_terrain.py --terrain stepping_stones

输出：/tmp/terrain_<name>.png（2D 高度场热图）
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--terrain", type=str, default="pyramid_stairs",
                    choices=["random_uniform", "pyramid_sloped", "pyramid_stairs",
                             "discrete_obstacles", "wave", "stepping_stones"])
parser.add_argument("--size", type=float, nargs=2, default=[10.0, 10.0])
args = parser.parse_args()
app = AppLauncher(args).app

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from isaaclab.terrains.height_field import hf_terrains, hf_terrains_cfg

# 每个地形类型 → (config class, 构造参数)
BASE = dict(size=tuple(args.size), horizontal_scale=0.1, vertical_scale=0.005, border_width=0.5)
TERRAINS = {
    "random_uniform": (hf_terrains_cfg.HfRandomUniformTerrainCfg,
                       {**BASE, "noise_range": (-0.05, 0.05), "noise_step": 0.005, "downsampled_scale": 0.2}),
    "pyramid_sloped": (hf_terrains_cfg.HfPyramidSlopedTerrainCfg,
                       {**BASE, "slope_range": (0.0, 0.4), "platform_width": 1.5}),
    "pyramid_stairs": (hf_terrains_cfg.HfPyramidStairsTerrainCfg,
                       {**BASE, "step_height_range": (0.05, 0.15), "step_width": 0.3, "platform_width": 1.5}),
    "discrete_obstacles": (hf_terrains_cfg.HfDiscreteObstaclesTerrainCfg,
                           {**BASE, "obstacle_height_mode": "choice", "num_obstacles": 20}),
    "wave": (hf_terrains_cfg.HfWaveTerrainCfg,
             {**BASE, "amplitude_range": (0.05, 0.2), "num_waves": 4}),
    "stepping_stones": (hf_terrains_cfg.HfSteppingStonesTerrainCfg,
                        {**BASE, "stone_height_range": (0.05, 0.2), "stone_size": 0.3, "stone_distance": 0.1}),
}

cfg_cls, kwargs = TERRAINS[args.terrain]
cfg = cfg_cls(**kwargs)
# cfg.function 被 @height_field_to_mesh 装饰（返回 meshes）；.__wrapped__ 拿原始 2D 高度场
func = cfg.function
raw = func.__wrapped__(difficulty=1.0, cfg=cfg)
hf = raw * cfg.vertical_scale  # 离散单位 → 米

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(hf.T, origin="lower", cmap="terrain", extent=[0, cfg.size[0], 0, cfg.size[1]])
ax.set_title(f"{args.terrain}  shape={hf.shape}  z∈[{hf.min():.2f},{hf.max():.2f}] m")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal")
fig.colorbar(im, ax=ax, label="height (m)")
out = f"/tmp/terrain_{args.terrain}.png"
fig.savefig(out, dpi=110)
print(f"saved {out}", flush=True)
os._exit(0)
