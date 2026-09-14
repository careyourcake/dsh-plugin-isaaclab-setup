#!/usr/bin/env python3
"""传感器验证模板：IMU + 接触力 + 高度扫描(RayCaster) + 深度相机，挂在 Go2 上并读数据。

用法：python sensors_demo.py --headless --enable_cameras
输出：各传感器的 shape / 数值范围，验证挂载正确（sensor 是 RL 环境的一部分）。

挂载参考标准 rough env（isaaclab_tasks/.../velocity_env_cfg.py）：
  height_scanner: RayCaster 从 base 上方 20m 往下打网格射线，打向 /World/ground
  contact_forces: ContactSensor 挂 Robot/.*
  imu:            ImuCfg 挂 Robot/base
  camera:         CameraCfg 挂 Robot/base/front_cam
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, ImuCfg, RayCasterCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, MeshPyramidStairsTerrainCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0), border_width=1.0, num_rows=1, num_cols=1,
            horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75,
            sub_terrains={"stairs": MeshPyramidStairsTerrainCfg(
                proportion=1.0, step_height_range=(0.12, 0.18), step_width=0.35,
                platform_width=1.5, border_width=1.0, holes=False)},
        ),
        visual_material=None, debug_vis=False,
    )
    dome_light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1.0))
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # ---- 传感器 ----
    imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/base")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),   # 从上方 20m 往下打
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
        update_period=0, height=240, width=320, data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, focus_distance=400.0,
                                         horizontal_aperture=20.955, clipping_range=(0.1, 20.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.4, 0.0, 0.05), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    print("sensors in scene:", list(scene.sensors.keys()), flush=True)

    default = scene["robot"].data.default_joint_pos.clone()
    scene["robot"].write_root_pose_to_sim(
        torch.tensor([[0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0]], device=args.device))
    for _ in range(300):
        scene["robot"].set_joint_position_target(default)
        scene.write_data_to_sim()
        sim.step()
        scene.update(0.005)

    imu = scene["imu"].data
    ct = scene["contact_forces"].data
    hs = scene["height_scanner"].data
    cam = scene["camera"].data
    print(f"[IMU]        lin_acc_b={tuple(imu.lin_acc_b.shape)} ang_vel_b={tuple(imu.ang_vel_b.shape)} "
          f"grav={tuple(imu.projected_gravity_b.shape)}", flush=True)
    print(f"[Contact]    net_forces_w_history={tuple(ct.net_forces_w_history.shape)} "
          f"air_time={tuple(ct.last_air_time.shape) if ct.last_air_time is not None else None}", flush=True)
    print(f"[RayCaster]  ray_hits_w={tuple(hs.ray_hits_w.shape)} "
          f"z∈[{hs.ray_hits_w[...,2].min():.2f},{hs.ray_hits_w[...,2].max():.2f}]", flush=True)
    print(f"[Camera]     rgb={tuple(cam.output['rgb'].shape)} "
          f"depth∈[{cam.output['distance_to_image_plane'].min():.2f},"
          f"{cam.output['distance_to_image_plane'].max():.2f}]", flush=True)
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
