import numpy as np

# Configuration constants
sim_duration = 20.
dt = 0.005
decimation = 20

mujoco_model_path = 'urdf/a1/xml/a1.xml'

unit_gvec = np.array([0., 0., -1.])

kp_sim = np.array([20., 20., 20.,
                   20., 20., 20.,
                   20., 20., 20.,
                   20., 20., 20.], dtype=np.double)

kd_sim = np.array([.5, .5, .5,
                   .5, .5, .5,
                   .5, .5, .5,
                   .5, .5, .5], dtype=np.double)

alter_index = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]

default_q_sim = np.array([
     0.1, 0.8, -1.5,
    -0.1, 0.8, -1.5,
     0.1, 1.0, -1.5,
    -0.1, 1.0, -1.5], dtype=np.double)

target_q_sim = default_q_sim.copy()
target_dq_sim = np.zeros_like(target_q_sim)

tau_limit_sim = np.array([
    23.7, 23.7, 35.55,
    23.7, 23.7, 35.55,
    23.7, 23.7, 35.55,
    23.7, 23.7, 35.55], dtype=np.double)

cmd_vx = 1.0
cmd_vy = 0.0
cmd_dyaw = 0.0

action_scale = 0.25
scale_lin_vel = 2.0
scale_ang_vel = 0.25
scale_joint_pos = 1.0
scale_joint_vel = 0.05