import time
import matplotlib.pyplot as plt
import numpy as np
import cv2

import tellox as tx

# Constants/environment definition
MAX_HEIGHT = 2  # meters; depends on room, used for safety checks
MAX_VEL_MAG = 0.2  # m/s; for safety
DELTA_POS = 0.1  # increments for sending velocity commands
VEL_CONTROL_RATE = 10.0  # Hz
MAX_FLIGHT_TIME = 300  # seconds
GATE_NUM_TAGS = 4  # defines how many AprilTags make up a full gate

# Control law values
X_REF = -3  # meters; for stabilizing drone relative to AprilTag gate

# Stabilization thresholds
X_THRESH = 0.1  # meters
Y_THRESH = 0.1
Z_THRESH = 0.1

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
    distance_camera_frame = gate_center_camera_frame - poses_camera_frame[idx, :]
    gate_center_y_dist = distance_camera_frame[0]
    gate_center_z_dist = distance_camera_frame[1]
    drone_pose_global, _, _ = pilot.get_drone_pose(tags[top_left_idx])
    return np.array([drone_pose_global[0], drone_pose_global[1] + gate_center_y_dist, drone_pose_global[2] + gate_center_z_dist])


def get_top_left_idx(tags: list):
    """
    TODO
    """
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


if __name__ == "__main__":
    print("Starting script...")
    # For tracking flight time
    start_time = time.time()

    # Collection containers for sensor readings
    altitudes = []
    times = []
    controls_altitude = []

    # Make a pilot object and take off
    print("Drone taking off")
    pilot = tx.Pilot()
    pilot.takeoff()

    # We're assuming that the drone starts facing the gate, so we don't need to rotate to search
    gate_found = False
    while not gate_found:
        # Log sensor readings
        readings = pilot.get_sensor_readings()
        altitudes.append(readings.height)
        times.append(readings.flight_time)

        # Current height
        z = altitudes[-1]
        if z > MAX_HEIGHT:
            print("WARNING: Flew too high! Landing for safety...")
            pilot.land()
            break

        # Search for tags
        img = pilot.get_camera_frame(visualize=False)
        tags = pilot.detect_tags(img, visualize=True)
        if tags:
            print("Detected {} AprilTags in image!".format(len(tags)))
            # TODO we could slow down the velocity here
            # Stop only when all tags are in FOV
            # TODO IDing tags is fussy, so this may not trigger perfectly
            if len(tags) == GATE_NUM_TAGS:
                print("Gate found in field of view!")
                gate_found = True
                # Stop increasing altitude
                pilot.send_control(np.zeros(3), 0.0)
        else:
            # Increase altitude to keep searching
            pilot.send_control(xyz_velocity, yaw_velocity)
            print("Increasing altitude...")
            # vz = K1 * (z - Z_REF) TODO
            vz = DELTA_POS
            vz = np.clip(vz, -1 * MAX_VEL_MAG, MAX_VEL_MAG)
            xyz_velocity = np.array([0.0, 0.0, vz])
            pilot.send_control(xyz_velocity, 0.0)  # no yaw
            controls_altitude.append(xyz_velocity)
            time.sleep(1 / VEL_CONTROL_RATE)

        # Land if exceeded max flight time
        if (time.time() - start_time) > MAX_FLIGHT_TIME:
            print("Exceeded max flight time of {} without detecting AprilTag gate.".format(MAX_FLIGHT_TIME))
            print("Landing...")
            pilot.land()
            break

    print("Stabilizing at gate center...")

    controls_gate = []  # to store velocity control commands
    positions = []
    stabilized = False
    while not stabilized:
        # Log sensor readings
        readings = pilot.get_sensor_readings()
        altitudes.append(readings.height)
        times.append(readings.flight_time)

        # Detect new tag positions
        img = pilot.get_camera_frame(visualize=False)
        tags = pilot.detect_tags(img, visualize=True)
        print("Detected {} AprilTags in FOV; expected {}".format(len(tags), GATE_NUM_TAGS))
        # TODO just keep looping if it doesn't work?
        if len(tags) < GATE_NUM_TAGS:
            print("Waiting to re-detect all tags...")
            continue

        # Use the top-left gate tag as the reference point
        reference_tag_idx = get_top_left_idx(tags)
        gate_center_pos = get_gate_center(pilot, tags, reference_tag_idx)
        print("Gate center wrt tag ID {} = {}".format(reference_tag_idx, gate_center_pos))

        position , _, _ = pilot.get_drone_pose(tags[reference_tag_idx])
        print("Current position wrt reference tag:", position)
        positions.append(position)

        # Check stabilization thresholds, then land the drone if within bounds
        x_diff = position[0] - X_REF  # want to be this far away from gate center
        y_diff = position[1] - gate_center_pos[1]
        z_diff = position[2] - gate_center_pos[2]
        if abs(x_diff) <= X_THRESH and abs(y_diff) <= Y_THRESH and abs(z_diff) <= Z_THRESH:
            stabilized = True
            # TODO land just for now
            pilot.land()
            cv2.imwrite("GateVisible.png", img)
            break

        else:
            # Stabilize at the desired point
            # TODO replace with actual controllers
            # TODO I think we actually don't want x_vel commands? Assuming the drone can hold position okay
            x_vel = KXYZ * x_diff
            y_vel = KXYZ * y_diff
            z_vel = KXYZ * z_diff
            controls_gate.append((x_vel, y_vel, z_vel))

            # Make sure to not go over control command bounds
            x_vel = np.clip(x_vel, -1 * MAX_VEL_MAG, MAX_VEL_MAG)
            y_vel = np.clip(y_vel, -1 * MAX_VEL_MAG, MAX_VEL_MAG)
            z_vel = np.clip(z_vel, -1 * MAX_VEL_MAG, MAX_VEL_MAG)

            xyz_velocity = np.array([x_vel, y_vel, z_vel])
            # TODO eventually we want to plot what's actually sent, not computed
            # so should store these velocity commands instead
            print("Sending stabilization control:")
            print("Velocity: ", xyz_velocity)
            pilot.send_control(xyz_velocity, 0.0)  # no yaw
            # Let it run for a short amount of time before stopping?
            time.sleep(1 / VEL_CONTROL_RATE)
            pilot.send_control(np.array([0.0, 0.0, 0.0]), 0.0)


    # For debugging/analysis:
    # Plot logged data
    positions = np.array(positions)
    controls_altitude = np.array(controls_altitude)
    controls_gate = np.array(controls_gate)

    plt.figure()
    plt.plot(controls_altitude[:, 0], label="x_vel")
    plt.plot(controls_altitude[:, 1], label="y_vel")
    plt.plot(controls_altitude[:, 2], label="z_vel")
    plt.ylabel("Velocity control (m/s)")
    plt.title("Velocity control to detect gate")
    plt.legend()
    plt.savefig("controls_altitude.png")

    plt.figure()
    plt.plot(controls_gate[:, 0], label="x_vel")
    plt.plot(controls_gate[:, 1], label="y_vel")
    plt.plot(controls_gate[:, 2], label="z_vel")
    plt.ylabel("Velocity control (m/s)")
    plt.title("Velocity control to center")
    plt.legend()
    plt.savefig("controls_gate.png")

    plt.figure()
    plt.plot(positions[:, 0], label="x")
    plt.plot(positions[:, 1], label="y")
    plt.plot(positions[:, 2], label="z")
    plt.title("Relative position of drone in top-left AprilTag frame")
    plt.ylabel("Rel. position (m)")
    plt.legend()
    plt.savefig("position_xyz.png")
