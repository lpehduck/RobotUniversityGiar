from legged_gym import *
import numpy as np
import torch
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math_utils import torch_rand_float
from legged_gym.utils.math_utils import *
from collections import deque

class G1Mimic(LeggedRobot):
    
    def compute_observations(self):
        frame_portions = (self.frame_idx / self.motion_length).unsqueeze(-1)
        ref_projected_gravity = quat_rotate_inverse(
            self.ref_root_rot[self.frame_idx], self.simulator._global_gravity)
        obs_buf = torch.cat((
            # proprioceptive features
            self.simulator.base_lin_vel * self.obs_scales.lin_vel,
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,
            self.simulator.projected_gravity,
            (self.simulator.dof_pos - 
             self.simulator.default_dof_pos) * self.obs_scales.dof_pos,
            self.simulator.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            # reference motion features
            frame_portions,
            self.ref_root_lin_vel[self.frame_idx] * self.obs_scales.lin_vel,
            self.ref_root_ang_vel[self.frame_idx] * self.obs_scales.ang_vel,
            ref_projected_gravity,
            (self.ref_dof_pos[self.frame_idx] - 
             self.simulator.default_dof_pos) * self.obs_scales.dof_pos,
            self.ref_dof_vel[self.frame_idx] * self.obs_scales.dof_vel,
        ), dim=-1)
        
        single_critic_obs = torch.cat((
            (self.simulator.base_pos - self.simulator.env_origins),
            (self.ref_root_pos[self.frame_idx] - self.simulator.env_origins),
            obs_buf
        ), dim=-1)
        
        self.critic_obs_deque.append(single_critic_obs)
        self.privileged_obs_buf = torch.cat(
            [self.critic_obs_deque[i]
                for i in range(self.critic_obs_deque.maxlen)],
            dim=-1,
        )
        
        if self.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec

        # push obs_buf to obs_history
        self.obs_history_deque.append(obs_buf)
        self.obs_buf = torch.cat(
            [self.obs_history_deque[i]
                for i in range(self.obs_history_deque.maxlen)],
            dim=-1,
        )
        
    def post_physics_step(self):
        self.frame_idx += 1
        # find out which envs have exceeded the motion length, and mark them as time out
        # resample frame_idx for the time out envs and reset root states and dofs of them seperately
        time_out_buf = self.frame_idx >= self.motion_length
        time_out_env_ids = time_out_buf.nonzero(as_tuple=False).flatten()
        self.frame_idx[time_out_env_ids] = torch.randint(0, self.motion_length, (len(time_out_env_ids),), device=self.device)
        self._reset_dofs(time_out_env_ids)
        self._reset_root_states(time_out_env_ids)
        super().post_physics_step()
        
    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        # randomly sample a starting frame index for each environment
        self.frame_idx[env_ids] = torch.randint(0, self.motion_length, (len(env_ids),), device=self.device)
        super().reset_idx(env_ids)
        # clear obs history for the envs that are reset
        for i in range(self.obs_history_deque.maxlen):
            self.obs_history_deque[i][env_ids] *= 0
        for i in range(self.critic_obs_deque.maxlen):
            self.critic_obs_deque[i][env_ids] *= 0
    
    def _reset_dofs(self, env_ids):
        # reset dofs to match the reference motion at the current frame index
        cur_frame_idx = self.frame_idx[env_ids]
        dof_pos = self.ref_dof_pos[cur_frame_idx]
        dof_vel = self.ref_dof_vel[cur_frame_idx]
        self.simulator.reset_dofs(env_ids, 
                                  dof_pos, 
                                  dof_vel)
    
    def _reset_root_states(self, env_ids):
        # reset root states to match the reference motion at the current frame index
        cur_frame_idx = self.frame_idx[env_ids]
        root_pos = self.ref_root_pos[cur_frame_idx] + self.simulator.env_origins[env_ids]
        root_pos[:, 2] += 0.1 # add a small vertical offset to avoid initial penetration
        root_rot = self.ref_root_rot[cur_frame_idx]
        root_lin_vel = torch.zeros_like(root_pos)
        root_ang_vel = torch.zeros_like(root_pos)
        self.simulator.reset_root_states(env_ids, 
                                         root_pos, 
                                         root_rot, 
                                         root_lin_vel, 
                                         root_ang_vel)
    
    def _get_noise_scale_vec(self):
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:6 + self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[6 + self.num_actions:6 + 2 * self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[6 + 2 * self.num_actions:6 + 3 * self.num_actions] = 0.  # previous actions
        noise_vec[6 + 3 * self.num_actions:7 + 3 * self.num_actions] = 0. # frame_portion
        noise_vec[7 + 3 * self.num_actions:10 + 3 * self.num_actions] = 0.  # ref_root_lin_vel
        noise_vec[10 + 3 * self.num_actions:13 + 3 * self.num_actions] = 0.  # ref_root_ang_vel
        noise_vec[13 + 3 * self.num_actions:16 + 3 * self.num_actions] = 0.  # ref_projected_gravity
        noise_vec[16 + 3 * self.num_actions:16 + 4 * self.num_actions] = 0.  # ref_dof_pos
        noise_vec[16 + 4 * self.num_actions:16 + 5 * self.num_actions] = 0.  # ref_dof_vel
        return noise_vec
    
    def _init_buffers(self):
        super()._init_buffers()
        # load reference motion data from the motion file specified in the config
        import pickle
        # Compatibility shim: pickle files saved with NumPy >= 2.0 reference
        # numpy._core (e.g. numpy._core.multiarray), which does not exist in
        # NumPy < 2.0.  Register aliases so unpickling works transparently.
        if not hasattr(np, '_core'):
            import numpy.core as _np_core
            for _attr in dir(_np_core):
                _mod_name = f"numpy._core.{_attr}"
                _submod = getattr(_np_core, _attr, None)
                if isinstance(_submod, type(sys)):
                    sys.modules.setdefault(_mod_name, _submod)
            sys.modules.setdefault("numpy._core", _np_core)
            sys.modules.setdefault("numpy._core.multiarray", _np_core.multiarray)
            del _np_core, _attr, _mod_name, _submod
        
        # open the .pkl file and load the motion data
        # the motion data should be a dictionary with the following keys:
        # motion_data = {
        #     "fps": aligned_fps,
        #     "root_pos": root_pos.cpu().numpy(),
        #     "root_lin_vel": root_lin_vel.cpu().numpy(),
        #     "root_rot": root_rot.cpu().numpy(),
        #     "root_euler": root_euler.cpu().numpy(),
        #     "root_ang_vel": root_ang_vel.cpu().numpy(),
        #     "dof_pos": dof_pos.cpu().numpy(),
        #     "dof_vel": dof_vel.cpu().numpy(),
        #     "key_body_pos_relative_to_base": key_body_pos_relative_to_base.cpu().numpy(),
        # }
        with open(self.cfg.env.motion_file, "rb") as f:
            motion_data = pickle.load(f)
        root_pos = motion_data["root_pos"]
        self.ref_root_pos = torch.from_numpy(root_pos).to(self.device).float()
        # judge if root_lin_vel is provided in the motion data, for visualing the reference motion from GMR
        if "root_lin_vel" in motion_data:
            root_lin_vel = motion_data["root_lin_vel"]
            self.ref_root_lin_vel = torch.from_numpy(root_lin_vel).to(self.device).float()
        else:
            self.ref_root_lin_vel = torch.zeros_like(self.ref_root_pos)
        root_rot = motion_data["root_rot"]
        self.ref_root_rot = torch.from_numpy(root_rot).to(self.device).float()
        if "root_ang_vel" in motion_data:
            root_ang_vel = motion_data["root_ang_vel"]
            self.ref_root_ang_vel = torch.from_numpy(root_ang_vel).to(self.device).float()
        else:
            self.ref_root_ang_vel = torch.zeros_like(self.ref_root_pos)
        dof_pos = motion_data["dof_pos"]
        self.ref_dof_pos = torch.from_numpy(dof_pos).to(self.device).float()
        if "dof_vel" in motion_data:
            dof_vel = motion_data["dof_vel"]
            self.ref_dof_vel = torch.from_numpy(dof_vel).to(self.device).float()
        else:
            self.ref_dof_vel = torch.zeros_like(self.ref_dof_pos)
        if "key_body_pos_relative_to_base" in motion_data:
            key_body_pos_relative_to_base = motion_data["key_body_pos_relative_to_base"]
            self.ref_key_body_pos_relative_to_base = torch.from_numpy(key_body_pos_relative_to_base).to(self.device).float()
        else:
            self.ref_key_body_pos_relative_to_base = torch.zeros_like(self.simulator.key_body_pos)
        self.motion_length = self.ref_root_pos.shape[0]
        assert self.ref_root_pos.shape[0] == self.ref_root_rot.shape[0] == self.ref_root_lin_vel.shape[0] == self.ref_root_ang_vel.shape[0] == self.ref_dof_pos.shape[0] == self.ref_dof_vel.shape[0], "Reference motion data length mismatch among different features"
        print(f"Loaded reference motion from {self.cfg.env.motion_file}, motion length: {self.motion_length} frames")
        # create a buffer to store the current frame index for each environment
        self.frame_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # flag set each step: True for envs whose frame_idx would exceed motion_length
        self._timed_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # obs_history
        self.obs_history_deque = deque(maxlen=self.cfg.env.frame_stack)
        for _ in range(self.cfg.env.frame_stack):
            self.obs_history_deque.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.num_single_obs,
                    dtype=torch.float,
                    device=self.device,
                )
            )
        # critic observation buffer
        self.critic_obs_deque = deque(maxlen=self.cfg.env.c_frame_stack)
        for _ in range(self.cfg.env.c_frame_stack):
            self.critic_obs_deque.append(
                torch.zeros(
                    self.num_envs,
                    self.cfg.env.num_single_critic_obs,
                    dtype=torch.float,
                    device=self.device,
                )
            )
        
    def _reward_tracking_ref_dof_pos(self):
        """Reward term for imitating the reference motion's dof positions.
        """
        dof_pos_error = torch.sum(torch.square(
            self.simulator.dof_pos - 
            self.ref_dof_pos[self.frame_idx]), dim=-1)
        
        return torch.exp(-dof_pos_error 
                         / self.cfg.rewards.tracking_dof_pos_sigma)
    
    def _reward_tracking_ref_dof_vel(self):
        """Reward term for imitating the reference motion's dof velocities.
        """
        dof_vel_error = torch.sum(torch.square(
            self.simulator.dof_vel - 
            self.ref_dof_vel[self.frame_idx]), dim=-1)
        
        return torch.exp(-dof_vel_error 
                         / self.cfg.rewards.tracking_dof_vel_sigma)
        
    def _reward_tracking_ref_root_pose(self):
        """Reward term for imitating the reference motion's root position and orientaion.
        """
        root_pos_error = torch.sum(torch.square(
            self.simulator.base_pos - 
            (self.ref_root_pos[self.frame_idx] + 
             self.simulator.env_origins)), dim=-1)
        
        root_rot_error = torch.sum(torch.square(
            self.simulator.base_quat - 
            self.ref_root_rot[self.frame_idx]), dim=-1)
        
        return torch.exp(-(root_pos_error + 0.1 * root_rot_error) /
                         self.cfg.rewards.tracking_ref_root_pose_sigma)
    
    def _reward_tracking_ref_root_vel(self):
        """Reward term for imitating the reference motion's root linear velocity and root angular velocity.
        """
        root_lin_vel_error = torch.sum(torch.square(
            self.simulator.base_lin_vel - 
            self.ref_root_lin_vel[self.frame_idx]), dim=-1)
        
        root_ang_vel_error = torch.sum(torch.square(
            self.simulator.base_ang_vel - 
            self.ref_root_ang_vel[self.frame_idx]), dim=-1)
        
        return torch.exp(-(root_lin_vel_error + 0.1 * root_ang_vel_error) /
                         self.cfg.rewards.tracking_ref_root_vel_sigma)
    
    def _reward_tracking_ref_key_pos(self):
        """Reward term for imitating the reference motion's key body position relative to base.
        """
        key_body_pos_relative_to_base = self.simulator.key_body_pos - self.simulator.base_pos.unsqueeze(1)
        key_body_pos_error = torch.sum(torch.square(
            key_body_pos_relative_to_base - 
            self.ref_key_body_pos_relative_to_base[self.frame_idx]), dim=[1,2])
        
        return torch.exp(-key_body_pos_error / 
                         self.cfg.rewards.tracking_ref_key_pos_sigma)