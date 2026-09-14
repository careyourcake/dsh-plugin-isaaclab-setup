#!/usr/bin/env python3
"""3D 场景渲染模板：地形 + Go2 + 第三视角 RGB + 右下角 Depth Image 内嵌 + 步态动画 → GIF。

用法（headless 服务器）：
  python render_scene.py --headless --enable_cameras --terrain gap --steps 170

关键点（对照 skill 2.4）：
  * Camera 必须放进 InteractiveScene（scene["camera"]）；standalone 会挂
  * DomeLightCfg(intensity=1.0) 才正常；2000+ 反而全黑
  * DistantLightCfg 做主光源才有方向阴影
  * 深度相机输出 (H,W,1)，合成前必须 reshape
  * 机器人 PD 保持：set_joint_position_target(default)；索引 target[0, idx]
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--terrain", type=str, default="gap", choices=["gap", "stairs", "wave"])
parser.add_argument("--out", type=str, default="/tmp/scene")
parser.add_argument("--steps", type=int, default=170)
args = parser.parse_args()
app = AppLauncher(args).app

import math
import os

import numpy as np
import torch
from PIL import Image, ImageDraw
import imageio

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.terrains import (
    TerrainImporterCfg, TerrainGeneratorCfg,
    MeshGapTerrainCfg, MeshPyramidStairsTerrainCfg, HfWaveTerrainCfg,
)
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

ZOO = {
    "gap": MeshGapTerrainCfg(proportion=1.0, gap_width_range=(1.5, 1.5), platform_width=4.0),
    "stairs": MeshPyramidStairsTerrainCfg(proportion=1.0, step_height_range=(0.08, 0.14),
                                          step_width=0.3, platform_width=2.0, border_width=1.0, holes=False),
    "wave": HfWaveTerrainCfg(proportion=1.0, amplitude_range=(0.05, 0.15), num_waves=4, border_width=0.25),
}


@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0), border_width=1.0, num_rows=1, num_cols=1,
            horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75,
            sub_terrains={args.terrain: ZOO[args.terrain]},
        ),
        visual_material=None, debug_vis=False,
    )
    dome_light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1.0))
    sun = AssetBaseCfg(
        prim_path="/World/Sun",
        spawn=sim_utils.DistantLightCfg(intensity=8.0, color=(1.0, 0.97, 0.92), angle=1.5),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.87, 0.0, 0.5, 0.0)),
    )
    floor = AssetBaseCfg(
        prim_path="/World/Floor",
        spawn=sim_utils.CuboidCfg(size=(30.0, 30.0, 0.1),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.06, 0.06, 0.07))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -2.0)),
    )
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/main_cam", update_period=0, height=720, width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0,
                                         horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )
    depth_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/depth_cam", update_period=0, height=300, width=300,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=20.0, focus_distance=400.0,
                                         horizontal_aperture=20.955, clipping_range=(0.1, 50.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )


def composite(rgb, depth):
    d = np.clip(depth.reshape(depth.shape[0], depth.shape[1]), 0.0, 8.0) / 8.0
    depth_rgb = np.stack([(255 * (1.0 - d)).astype(np.uint8)] * 3, axis=-1)
    main = Image.fromarray((rgb * 255).astype(np.uint8))
    inset = Image.fromarray(depth_rgb).resize((300, 300))
    W, H = main.size
    x0, y0 = W - 320, H - 350
    draw = ImageDraw.Draw(main)
    draw.rectangle([x0, y0, x0 + 320, y0 + 350], fill=(255, 255, 255))
    main.paste(inset, (x0 + 10, y0 + 40))
    draw.text((x0 + 10, y0 + 12), "Depth Image", fill=(0, 0, 0))
    return np.asarray(main)


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    scene["camera"].set_world_poses_from_view(
        torch.tensor([[0.1, 2.2, 1.55]], device=args.device),
        torch.tensor([[0.1, -0.3, 0.05]], device=args.device))
    scene["depth_cam"].set_world_poses_from_view(
        torch.tensor([[0.2, 1.2, 3.2]], device=args.device),
        torch.tensor([[0.2, -0.4, 0.0]], device=args.device))

    names = list(scene["robot"].joint_names)
    idx = {n: i for i, n in enumerate(names)}
    default = scene["robot"].data.default_joint_pos.clone()
    thigh = {L: idx[f"{L}_thigh_joint"] for L in ["FL", "FR", "RL", "RR"]}
    calf = {L: idx[f"{L}_calf_joint"] for L in ["FL", "FR", "RL", "RR"]}
    phase = {"FL": 0.0, "RR": 0.0, "FR": math.pi, "RL": math.pi}

    dt, freq, speed = 0.005, 1.6, 1.1
    x_base, frames = -0.35, []
    for f in range(args.steps):
        t = f * dt
        target = default.clone()
        for L in ["FL", "FR", "RL", "RR"]:
            target[0, thigh[L]] = default[0, thigh[L]] + 0.5 * math.sin(2 * math.pi * freq * t + phase[L])
            target[0, calf[L]] = default[0, calf[L]] - 0.35 * max(0.0, math.cos(2 * math.pi * freq * t + phase[L]))
        x_base += speed * dt
        pos = torch.tensor([[x_base, 0.0, 0.36]], device=args.device)
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=args.device)
        scene["robot"].write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
        scene["robot"].set_joint_position_target(target)
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)
        if f % 10 == 0:
            scene["camera"].update(dt=0.0)
            scene["depth_cam"].update(dt=0.0)
            rgb = scene["camera"].data.output["rgb"][0, ..., :3].cpu().numpy()
            depth = scene["depth_cam"].data.output["distance_to_image_plane"][0].cpu().numpy()
            frames.append(composite(rgb, depth))
    imageio.mimsave(f"{args.out}_walk.gif", frames, fps=4)
    print(f"saved {args.out}_walk.gif ({len(frames)} frames)", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
