#!/usr/bin/env python3
"""论文级 3D 渲染模板：RTX 光追 + PBR 材质 + 地形 + Go2 + 深度内嵌 + MP4/GIF。

用法（headless GPU 服务器）：
  python render_scene.py --headless --enable_cameras --terrain stairs --steps 300

渲染管线要点（对照 skill 4.4）：
  * RTX 光追：SimulationCfg(render=RenderCfg(...)) 开 GI/反射/软阴影/AO/DLAA
  * PBR 材质：地形 visual_material=PreviewSurfaceCfg(...)，别留默认灰
  * 1080p + MP4（imageio 走 ffmpeg）
  * Camera 走 InteractiveScene；DomeLight intensity=1.0 附近；低角度看台阶立面
  * 机器人 z 不写死：先 settle 读实际站高，相机随站高
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--terrain", type=str, default="stairs", choices=["gap", "stairs", "wave"])
parser.add_argument("--out", type=str, default="/tmp/scene")
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--res", type=str, default="1920x1080", help="主视角分辨率 WxH")
parser.add_argument("--shot", type=str, default="auto",
                    choices=["auto", "behind", "left", "back_diag", "front_3q"],
                    help="跟拍机位；auto = 按地形自动选（窄道 behind / 跳跃 left / 坡道 back_diag）")
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
from isaaclab.sim import RenderCfg, SimulationCfg
from isaaclab.terrains import (
    TerrainImporterCfg, TerrainGeneratorCfg,
    MeshGapTerrainCfg, MeshPyramidStairsTerrainCfg, HfWaveTerrainCfg,
)
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

ZOO = {
    "gap": MeshGapTerrainCfg(proportion=1.0, gap_width_range=(1.5, 1.5), platform_width=4.0),
    "stairs": MeshPyramidStairsTerrainCfg(proportion=1.0, step_height_range=(0.14, 0.22),
                                          step_width=0.35, platform_width=1.5, border_width=1.0, holes=False),
    "wave": HfWaveTerrainCfg(proportion=1.0, amplitude_range=(0.05, 0.15), num_waves=4, border_width=0.25),
}

# ---- 任务自适应跟拍机位（相对机器狗本体系，机器人转相机跟着转）----
# 论文视频惯例：窄道跟正后方、跳跃跟正左、上下坡跟斜后方、展示用斜前方
SHOTS = {
    "behind":    ((-2.6, 0.0, 0.95), (0.9, 0.0, -0.35)),    # 正后方（窄道/直行）
    "left":      ((0.0, 2.6, 0.85), (0.0, 0.0, -0.35)),    # 正左（跳跃/侧向动作）
    "back_diag": ((-2.0, 1.7, 0.95), (0.6, 0.0, -0.35)),   # 斜后方（上下坡/楼梯）
    "front_3q":  ((2.0, 1.7, 0.95), (-0.5, 0.0, -0.35)),   # 斜前方（展示/迎面）
}
# 地形 → 默认机位（可被 --shot 覆盖）
TERRAIN_SHOT = {"gap": "left", "stairs": "back_diag", "wave": "back_diag"}

W, H = (int(x) for x in args.res.lower().split("x"))

# PBR 材质（混凝土感，别用默认灰）
TERRAIN_MAT = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.34, 0.33, 0.31), roughness=0.8, metallic=0.0)


@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0), border_width=1.0, num_rows=1, num_cols=1,
            horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75,
            sub_terrains={args.terrain: ZOO[args.terrain]},
        ),
        visual_material=TERRAIN_MAT,   # ← PBR 材质
        debug_vis=False,
    )
    dome_light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1.0))
    # 主光（太阳）——出方向阴影
    sun = AssetBaseCfg(
        prim_path="/World/Sun",
        spawn=sim_utils.DistantLightCfg(intensity=2.0, color=(1.0, 0.97, 0.93), angle=2.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.87, 0.0, 0.5, 0.0)),
    )
    # 补光（天光，冷色，压低反差更像影棚）
    fill = AssetBaseCfg(
        prim_path="/World/Fill",
        spawn=sim_utils.DistantLightCfg(intensity=0.6, color=(0.95, 0.95, 1.0), angle=5.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.95, 0.25, -0.2, 0.0)),
    )
    floor = AssetBaseCfg(
        prim_path="/World/Floor",
        spawn=sim_utils.CuboidCfg(size=(30.0, 30.0, 0.1),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.06))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -2.0)),
    )
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/main_cam", update_period=0, height=H, width=W,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=28.0, focus_distance=400.0,
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


def composite(rgb, depth, inset=340):
    d = np.clip(depth.reshape(depth.shape[0], depth.shape[1]), 0.0, 8.0) / 8.0
    depth_rgb = np.stack([(255 * (1.0 - d)).astype(np.uint8)] * 3, axis=-1)
    main = Image.fromarray((rgb * 255).astype(np.uint8))
    im = Image.fromarray(depth_rgb).resize((inset, inset))
    w, h = main.size
    x0, y0 = w - inset - 30, h - inset - 40
    draw = ImageDraw.Draw(main)
    draw.rectangle([x0 - 10, y0 - 34, x0 + inset + 10, y0 + inset + 10], fill=(255, 255, 255))
    main.paste(im, (x0, y0))
    draw.text((x0, y0 - 26), "Depth Image", fill=(0, 0, 0))
    return np.asarray(main)


def cam_follow(scene, robot_pos, yaw, shot, device):
    """让相机相对机器狗本体系跟拍：机器人转，相机跟着转（任务自适应机位）。"""
    off, look = SHOTS[shot]
    c, s = math.cos(yaw), math.sin(yaw)

    def rot(v):
        return (c * v[0] - s * v[1], s * v[0] + c * v[1])

    px, py = rot(off)
    lx, ly = rot(look)
    rx, ry, rz = float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2])
    eye = torch.tensor([[rx + px, ry + py, rz + off[2]]], device=device, dtype=torch.float32)
    tgt = torch.tensor([[rx + lx, ry + ly, rz + look[2]]], device=device, dtype=torch.float32)
    scene["camera"].set_world_poses_from_view(eye, tgt)


def main():
    # RTX 光追管线：GI + 反射 + 软阴影 + AO + DLAA 抗锯齿
    sim_cfg = SimulationCfg(
        dt=0.005, device=args.device,
        render=RenderCfg(
            enable_reflections=True,
            enable_global_illumination=True,
            enable_shadows=True,
            enable_translucency=True,
            enable_ambient_occlusion=True,
            enable_direct_lighting=True,
            enable_dl_denoiser=True,
            antialiasing_mode="DLAA",
            samples_per_pixel=4,
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    # 地形材质：TerrainImporterCfg.visual_material 不一定生效 —— 用标准 USD 方式直接绑定
    try:
        import omni.usd
        from pxr import UsdShade, Sdf
        stage = omni.usd.get_context().get_stage()
        mat_path = "/World/Looks/TerrainMat"
        mat = UsdShade.Material.Define(stage, mat_path)
        sh = UsdShade.Shader.Define(stage, mat_path + "/PreviewSurface")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.38, 0.36, 0.34))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        prim = stage.GetPrimAtPath("/World/ground")
        if prim.IsValid():
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)
            print("bound terrain material via USD MaterialBindingAPI", flush=True)
        else:
            print("terrain prim /World/ground not found", flush=True)
    except Exception as e:
        print(f"usd material bind failed: {e}", flush=True)
    print(f"render: {W}x{H} RTX GI+reflections+AO+DLAA, terrain PBR material applied", flush=True)

    names = list(scene["robot"].joint_names)
    idx = {n: i for i, n in enumerate(names)}
    default = scene["robot"].data.default_joint_pos.clone()
    thigh = {L: idx[f"{L}_thigh_joint"] for L in ["FL", "FR", "RL", "RR"]}
    calf = {L: idx[f"{L}_calf_joint"] for L in ["FL", "FR", "RL", "RR"]}
    phase = {"FL": 0.0, "RR": 0.0, "FR": math.pi, "RL": math.pi}

    dt = 0.005
    # 机器狗 z 不能写死：高抛后 PD 保持，让它落到地形上，读实际站高
    scene["robot"].write_root_pose_to_sim(
        torch.tensor([[0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0]], device=args.device))
    for _ in range(400):
        scene["robot"].set_joint_position_target(default)
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)
    z_stand = float(scene["robot"].data.root_pos_w[0, 2]) + 0.06
    print(f"settled standing height z={z_stand:.3f}", flush=True)

    # 任务自适应跟拍机位
    shot = TERRAIN_SHOT.get(args.terrain, "back_diag") if args.shot == "auto" else args.shot
    print(f"camera shot = {shot} (terrain={args.terrain})", flush=True)
    # 深度相机：俯视整片地形（展示平台 + 间隙/台阶深度）
    scene["depth_cam"].set_world_poses_from_view(
        torch.tensor([[0.2, 1.2, z_stand + 3.4]], device=args.device),
        torch.tensor([[0.2, -0.4, z_stand - 0.4]], device=args.device))

    freq, speed = 1.6, 0.85
    x_base, frames = -0.35, []
    for f in range(args.steps):
        t = f * dt
        target = default.clone()
        for L in ["FL", "FR", "RL", "RR"]:
            target[0, thigh[L]] = default[0, thigh[L]] + 0.5 * math.sin(2 * math.pi * freq * t + phase[L])
            target[0, calf[L]] = default[0, calf[L]] - 0.35 * max(0.0, math.cos(2 * math.pi * freq * t + phase[L]))
        x_base += speed * dt
        pos = torch.tensor([[x_base, 0.0, z_stand]], device=args.device)
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=args.device)
        scene["robot"].write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
        scene["robot"].set_joint_position_target(target)
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)
        if f % 6 == 0:
            # 相机跟拍：按机器狗当前位置 + 本体系机位偏移重定位
            rp = scene["robot"].data.root_pos_w[0].cpu().numpy()
            cam_follow(scene, rp, yaw=0.0, shot=shot, device=args.device)
            for _ in range(2):
                sim.step()
            scene.update(dt)
            scene["camera"].update(dt=0.0)
            scene["depth_cam"].update(dt=0.0)
            rgb = scene["camera"].data.output["rgb"][0, ..., :3].cpu().numpy()
            depth = scene["depth_cam"].data.output["distance_to_image_plane"][0].cpu().numpy()
            frames.append(composite(rgb, depth))

    try:
        imageio.mimsave(f"{args.out}_walk.mp4", frames, fps=25, quality=9)
        print(f"saved {args.out}_walk.mp4 ({len(frames)} frames)", flush=True)
    except Exception as e:
        print(f"mp4 failed: {e}", flush=True)
    imageio.mimsave(f"{args.out}_walk.gif", frames, fps=4)
    imageio.imwrite(f"{args.out}_frame.png", frames[len(frames) // 2])
    print(f"saved gif+frame ({len(frames)} frames)", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
