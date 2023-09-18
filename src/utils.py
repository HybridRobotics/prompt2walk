import numpy as np
from scipy.spatial.transform import Rotation as R

def compute_camera_position(agent_pos, distance=2.0, offset_angle=135):
    r = R.from_quat([0, 0, 0, 1])
    offset_rad = np.radians(offset_angle)
    dx = distance * np.cos(offset_rad)
    dy = distance * np.sin(offset_rad)
    rotated_offset = r.apply([dx, dy, 0])
    cam_pos = agent_pos.copy()
    cam_pos[2] = 0.0
    cam_pos += rotated_offset
    return cam_pos

def update_camera_position(data, viewer, cam_pos_func):
    agent_pos = data.qpos[:3]
    cam_pos = cam_pos_func(agent_pos)
    viewer.cam.lookat[:3] = agent_pos
    viewer.cam.lookat[-1] = 0.2
    viewer.cam.distance = np.linalg.norm(cam_pos - agent_pos)
    viewer.cam.azimuth = np.degrees(np.arctan2(cam_pos[1] - agent_pos[1], cam_pos[0] - agent_pos[0]))
    viewer.cam.elevation = -15  # adjust this based on the desired vertical angle

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd