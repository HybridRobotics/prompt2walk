import mujoco, mujoco_viewer
import numpy as np
from tqdm import tqdm
from ast import literal_eval

from config import sim_duration, dt, mujoco_model_path
from utils import compute_camera_position, update_camera_position

'''Simulation'''

model = mujoco.MjModel.from_xml_path(mujoco_model_path)
model.opt.timestep = dt
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
viewer = mujoco_viewer.MujocoViewer(model, data)

desc = 'Simulating...'

lines = []
with open('trajectory.txt', 'r') as f:
    for line in f:
        lines.append(line.strip())
counter = 0

for _ in tqdm(range(int(sim_duration / dt)), desc=desc):

    state = lines[counter]
    state = literal_eval(state)
    print(state)
    data.qpos[:] = state
    data.qvel[:] = np.zeros_like(data.qvel)
    counter += 1

    mujoco.mj_step(model, data)
    
    update_camera_position(data, viewer, compute_camera_position)

    viewer.render()
