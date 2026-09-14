#!/usr/bin/env python3
"""DirectRLEnv 完整模板：Go2 在可配置地形上行走（观测/奖励/终止齐全，可直接训练）。

复制本文件 → 改地形类型/奖励项 → 就是一个新环境。
用法：
  python direct_rl_env.py --headless --num-envs 64        # 跑通验证
  # 训练时把它包进 RslRlVecEnvWrapper（见 skill 3.1）

关键点（对照 skill 2.x）：
  * AppLauncher 必须先于 import isaaclab.sim
  * 地形由 TerrainImporterCfg 提供；机器人 prim 用 {ENV_REGEX_NS}/Robot
  * 向量化：所有 write_*_to_sim 不传 env_ids
  * 结尾 os._exit(0)（headless close 会卡）
"""
import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--terrain", type=str, default="stairs",
                    choices=["stairs", "gap", "wave", "random_rough", "pyramid_slope"])
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import (
    TerrainImporterCfg, TerrainGeneratorCfg,
    MeshPyramidStairsTerrainCfg, MeshGapTerrainCfg,
    HfWaveTerrainCfg, HfRandomUniformTerrainCfg, HfPyramidSlopedTerrainCfg,
)
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

TERRAIN_ZOO = {
    "stairs": MeshPyramidStairsTerrainCfg(proportion=1.0, step_height_range=(0.06, 0.14),
                                          step_width=0.3, platform_width=2.0, border_width=1.0, holes=False),
    "gap": MeshGapTerrainCfg(proportion=1.0, gap_width_range=(0.4, 0.7), platform_width=3.0),
    "wave": HfWaveTerrainCfg(proportion=1.0, amplitude_range=(0.02, 0.08), num_waves=4, border_width=0.25),
    "random_rough": HfRandomUniformTerrainCfg(proportion=1.0, noise_range=(0.02, 0.10),
                                              noise_step=0.02, border_width=0.25),
    "pyramid_slope": HfPyramidSlopedTerrainCfg(proportion=1.0, slope_range=(0.0, 0.3),
                                               platform_width=2.0, border_width=0.25),
}


@configclass
class Go2TerrainEnvCfg(DirectRLEnvCfg):
    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=4)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=4.0, replicate_physics=True)
    decimation = 4
    episode_length_s = 20.0
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,))
    observation_space = gym.spaces.Box(low=-100.0, high=100.0, shape=(48,))
    # 目标前进速度（速度跟踪奖励的目标）
    target_vx: float = 0.8


class Go2TerrainEnv(DirectRLEnv):
    cfg: Go2TerrainEnvCfg

    def __init__(self, cfg, render_mode=None):
        super().__init__(cfg, render_mode)
        self._actions = torch.zeros(self.num_envs, 12, device=self.device)
        self._commands = torch.full((self.num_envs,), cfg.target_vx, device=self.device)
        # 关节顺序：按 Go2 的 12 个 leg 关节
        self._joint_ids = [self.robot.joint_names.index(n) for n in self.robot.joint_names]
        self._action_scale = 0.25
        self._default_joint_pos = self.robot.data.default_joint_pos.clone()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact"] = self.contact_sensor
        self.scene.clone_environments(copy_from_source=False)

    def _pre_physics_step(self, actions):
        self._actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self):
        target = self._default_joint_pos + self._action_scale * self._actions
        self.robot.set_joint_position_target(target)

    def _get_observations(self):
        base_lin_vel = self.robot.data.root_lin_vel_b          # (N,3)
        base_ang_vel = self.robot.data.root_ang_vel_b          # (N,3)
        gravity = self.robot.data.projected_gravity_b          # (N,3)
        joint_pos = self.robot.data.joint_pos - self._default_joint_pos
        joint_vel = self.robot.data.joint_vel
        obs = torch.cat([base_lin_vel, base_ang_vel, gravity, joint_pos, joint_vel,
                         self._commands.unsqueeze(-1)], dim=-1)   # 3+3+3+12+12+1 = 34
        return {"policy": obs}

    def _get_rewards(self):
        vx = self.robot.data.root_lin_vel_b[:, 0]
        # 速度跟踪：指数核
        track = torch.exp(-((vx - self._commands) ** 2) / 0.25)
        # 罚项
        ang_vel = self.robot.data.root_ang_vel_b
        gravity = self.robot.data.projected_gravity_b
        joint_vel = self.robot.data.joint_vel
        action_rate = torch.sum((self._actions - self._prev_actions) ** 2, dim=-1) if hasattr(self, "_prev_actions") else torch.zeros_like(vx)
        rew = 1.5 * track \
              - 0.05 * torch.sum(ang_vel[:, :2] ** 2, dim=-1) \
              + 0.5 * gravity[:, 2] \
              - 0.001 * torch.sum(joint_vel ** 2, dim=-1) \
              - 0.01 * action_rate
        # 姿态/接触惩罚
        contact = self.contact_sensor.data.net_forces_w_history[:, -1].norm(dim=-1).max(dim=-1).values
        rew = rew - 1.0 * (contact > 1.0).float()
        self._prev_actions = self._actions.clone()
        return rew

    def _get_dones(self):
        base_contact = (self.contact_sensor.data.net_forces_w_history[:, -1].norm(dim=-1).max(dim=-1).values > 1.0)
        terminated = base_contact
        truncated = self.episode_length_buf > int(self.cfg.episode_length_s / (self.cfg.sim.dt * self.cfg.decimation))
        return terminated, truncated

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        # 随机初始速度指令
        self._commands[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(0.2, self.cfg.target_vx + 0.4)


# 地形 + 机器人 + 接触传感器配置（挂在 env cfg 上）
Go2TerrainEnvCfg.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
Go2TerrainEnvCfg.terrain = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
        size=(8.0, 8.0), border_width=1.0, num_rows=1, num_cols=1,
        horizontal_scale=0.1, vertical_scale=0.005, slope_threshold=0.75,
        sub_terrains={args.terrain: TERRAIN_ZOO[args.terrain]},
    ),
    visual_material=None, debug_vis=False,
)
Go2TerrainEnvCfg.contact_sensor = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, update_period=0.0, track_air_time=False,
)


def main():
    cfg = Go2TerrainEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = Go2TerrainEnv(cfg)
    obs, _ = env.reset()
    print(f"num_envs={cfg.scene.num_envs}  obs={obs['policy'].shape}", flush=True)
    # 随机动作跑几步，验证 obs/reward 无 NaN
    for i in range(20):
        act = torch.zeros(cfg.scene.num_envs, 12, device=env.device)
        obs, rew, term, trunc, _ = env.step(act)
        if i == 19:
            o = obs["policy"]
            print(f"step {i}: obs nan={torch.isnan(o).any().item()}  "
                  f"reward 范围=[{rew.min():.3f},{rew.max():.3f}]  terminated={term.sum().item()}", flush=True)
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
