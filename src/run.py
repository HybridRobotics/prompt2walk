import torch
import numpy as np
from tqdm import tqdm
import mujoco, mujoco_viewer
from ast import literal_eval
from decimal import Decimal
from llm import llm_init, llm_query
from scipy.spatial.transform import Rotation as R

import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

from config import sim_duration, dt, mujoco_model_path, decimation
from config import unit_gvec, kp_sim, kd_sim, alter_index, default_q_sim, target_q_sim, target_dq_sim, tau_limit_sim
from config import cmd_vx, cmd_vy, cmd_dyaw, action_scale, scale_lin_vel, scale_ang_vel, scale_joint_pos, scale_joint_vel

from utils import compute_camera_position, update_camera_position, pd_control

action = np.zeros_like(target_q_sim)
obs = np.zeros([1, 48], dtype=np.float32)

policy = torch.jit.load('policies/policy_1.pt')


'''Simulation'''

model = mujoco.MjModel.from_xml_path(mujoco_model_path)
model.opt.timestep = dt
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
viewer = mujoco_viewer.MujocoViewer(model, data)

def get_obs():
    '''Extracts an observation from the mujoco data structure
    '''
    # Order: [FRH, FRT, FRC,
    #         FLH, FLT, FLC,
    #         RRH, RRT, RRC,
    #         RLH, RLT, RLC]
    global data

    q = data.qpos.astype(np.double)[-12:]
    dq = data.qvel.astype(np.double)[-12:]
    quat = data.sensor('Body_Quat').data[[1, 2, 3, 0]].astype(np.double)
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.sensor('Body_Gyro').data.astype(np.double)
    gvec = r.apply(unit_gvec, inverse=True).astype(np.double)
    return (q, dq, quat, v, omega, gvec)

desc = 'Simulating...'
count = 0

llm_init()
msg = ""

output_file = open('trajectory.txt', 'w')

for _ in tqdm(range(int(sim_duration / dt)), desc=desc):
    # Obtain an observation
    q, dq, quat, v, omega, gvec = get_obs()
    q_sim, dq_sim = q[alter_index], dq[alter_index]

    state = ["{:.10f}".format(item) for item in data.qpos]
    output_file.write(f"{state}\n")

    if (count % decimation == 0) and (count >= 200):
    
        obs[0, 0:3] = data.qvel[:3] * scale_lin_vel
        obs[0, 3:6] = omega * scale_ang_vel
        obs[0, 6:9] = gvec
        obs[0, 9] = cmd_vx * scale_lin_vel
        obs[0, 10] = cmd_vy * scale_lin_vel
        obs[0, 11] = cmd_dyaw * scale_ang_vel
        obs[0, 12:24] = (q_sim - default_q_sim) * scale_joint_pos
        obs[0, 24:36] = dq_sim * scale_joint_vel
        obs[0, 36:48] = action

        printed_obs = [int((float(Decimal("%.1f" % x)) + 5) * 10) for x in obs[0]]
        printed_obs = printed_obs[0:9] + printed_obs[12:36]

        if msg != "":
            msg = msg + f"\nInput: {printed_obs}"
        else:
            msg = f"Input: {printed_obs}"

        action[:] = policy(torch.tensor(obs)).detach().numpy()
        action = np.clip(action, -5, 5)

        if count >= 2000:
            action_from_llm = llm_query(msg)
            msg = action_from_llm
            action = action_from_llm.split(": ", 1)[1]
            action = literal_eval(action)
            action = np.array([float(x) / 10.0 - 5.0 for x in action])
        else:
            if msg != "":
                llm_query(msg, call_api=False)
            printed_action = [int((float(Decimal("%.1f" % x)) + 5) * 10) for x in action]
            msg = f"Output: {printed_action}"

        target_q_sim = action * action_scale + default_q_sim

    count += 1

    # Generate PD control
    tau_sim = pd_control(target_q_sim, q_sim, kp_sim,
                         target_dq_sim, dq_sim, kd_sim)  # Calc torques
    tau_sim = np.clip(tau_sim, -tau_limit_sim, tau_limit_sim)  # Clamp torques
    tau = tau_sim[alter_index]

    data.ctrl = tau
    mujoco.mj_step(model, data)
    
    update_camera_position(data, viewer, compute_camera_position)

    viewer.render()
