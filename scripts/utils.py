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


def get_gate_center(pilot: tx.Pilot, tags: list, top_left_idx: int):
    """
    Takes list of Detection objects defining a gate and returns a numpy array denoting the
    center of the gate in the frame of the top-left tag at the given index
    """
    poses_camera_frame = np.zeros((len(tags), 3))
    for idx, t in enumerate(tags):
        poses_camera_frame[idx, :] = t.pose_t.reshape(3,)
    gate_center_camera_frame = np.mean(poses_camera_frame, axis=0)
    # Camera frame is +x right, +y down, +z forward, so the center will be positive distance from
    # the top-left tag of the gate
    distance_camera_frame = gate_center_camera_frame - poses_camera_frame[top_left_idx, :]
    print("Computed distance in camera frame = gate_center - top_left_pose_camera_frame:")
    print("\t", distance_camera_frame)
    gate_center_y_dist = distance_camera_frame[0]
    gate_center_z_dist = distance_camera_frame[1]
    # drone_pose_global, _, _ = pilot.get_drone_pose(tags[top_left_idx])
    # return np.array([drone_pose_global[0], drone_pose_global[1] + gate_center_y_dist, drone_pose_global[2] + gate_center_z_dist])
    return (gate_center_y_dist, gate_center_z_dist)


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


def get_top_left_tag_idx(tags: list):
    if len(tags) == 0:
        print("ERROR: Empty list of tags given!")
        return -1

    top_left_idx = 0
    leftmost_pos = tags[0].pose_t[0]  # +x to the right
    topmost_pos = tags[0].pose_t[1]  # +z down
    for idx, t in enumerate(tags):
        if t.pose_t[0] < leftmost_pos or t.pose_t[1] < topmost_pos:
            top_left_idx = idx
            leftmost_pos = t.pose_t[0]
            topmost_pos = t.pose_t[1]
    return top_left_idx