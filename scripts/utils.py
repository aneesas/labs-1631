import numpy as np
import time

import tellox as tx


def fly_open_loop(pilot, xyz_velocity, total_time, command_rate):
    time_elapsed = 0.0
    dt = 1.0/command_rate
    while time_elapsed < total_time:
        pilot.send_control(xyz_velocity, 0.0)
        time.sleep(dt)
        time_elapsed += dt
    pilot.send_control(np.array([0.0, 0.0, 0.0]), 0.0)
    print("Completed open-loop flight for {} seconds at {} m/s.".format(time_elapsed, xyz_velocity))
    return


def get_pose_gate_center(pilot: tx.Pilot, tags: list):
    """
    Takes list of Detection objects defining a gate and returns a numpy array denoting the
    center of the gate
    """
    drone_poses = np.zeros((len(tags), 3))
    for idx, t in enumerate(tags):
        drone_poses[idx, :], _, _ = pilot.get_drone_pose(t)
    pose_gate_center = np.mean(drone_poses, axis=0)
    return pose_gate_center


