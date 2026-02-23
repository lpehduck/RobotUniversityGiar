from legged_gym import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from legged_gym.envs.base.common_cfgs import G1MimicCommonCfg

class G1MimicCfg(G1MimicCommonCfg):
    class env(LeggedRobotCfg.env):
        frame_stack = 5
        num_single_obs = 164
        num_observations = int(num_single_obs * frame_stack)
        c_frame_stack = 5
        num_single_critic_obs = num_single_obs + 6
        num_privileged_obs = int(num_single_critic_obs * c_frame_stack)
        num_actions = 29
        # reference motion file, should be a .pkl file containing a dictionary with the following keys:
        motion_file = '/home/lupinjia/LeggedGym-Ex/resources/reference_motion/0005_walking001_stageii.pkl'
        clip_start = 0
        clip_end = -1
        episode_length_s = 10.0
        
    class domain_rand(G1MimicCommonCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 2.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.0

    class control(G1MimicCommonCfg.control):
        pass

    class asset(G1MimicCommonCfg.asset):
        pass

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        tracking_dof_pos_sigma = 4.0
        tracking_dof_vel_sigma = 100.0
        tracking_ref_root_pose_sigma = 0.2
        tracking_ref_root_vel_sigma = 1.0
        tracking_ref_key_pos_sigma = 0.1
        only_positive_rewards = False
        class scales(LeggedRobotCfg.rewards.scales):
            # # limits
            # dof_pos_limits = -5.0
            # tasks
            tracking_ref_dof_pos = 0.5
            tracking_ref_dof_vel = 0.1
            tracking_ref_root_pose = 0.15
            tracking_ref_root_vel = 0.1
            tracking_ref_key_pos = 0.15
            # keep_balance = 1.0
            # # regularization
            # lin_vel_z = -2.0
            # ang_vel_xy = -0.05
            # orientation = -0.5
            # dof_acc = -1.e-7
            # dof_power = -2.e-5
            # collision = -1.0
            # action_rate = -0.01
            # feet_slip = -1.0
            # action_smoothness = -0.01
    
    class normalization(LeggedRobotCfg.normalization):
        clip_actions = 10.0

class G1MimicCfgPPO(LeggedRobotCfgPPO):
    class policy(LeggedRobotCfgPPO.policy):
        clip_actions = G1MimicCfg.normalization.clip_actions
        init_noise_std = 0.8
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [1024, 256, 128]
        activation = 'elu'
    
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.0

    class runner(LeggedRobotCfgPPO.runner):
        max_iterations = 10000
        run_name = ''
        experiment_name = 'g1_mimic'
        save_interval = 500
