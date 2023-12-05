import time
import matplotlib.pyplot as plt
import numpy as np
import cv2

import tellox as tx

from utils import *

# Constants/environment definition
MAX_HEIGHT = 5  # meters; depends on room, used for safety checks
MAX_VEL_MAG = 0.3  # m/s; for safety
DELTA_POS = 0.5  # increments for sending velocity commands
VEL_CONTROL_RATE = 50.0  # Hz
MAX_FLIGHT_TIME = 100  # seconds
GATE_NUM_TAGS = 6  # defines how many AprilTags make up a full gate

# Control law values
X_REF = -1.5  # meters; for stabilizing drone relative to AprilTag gate
K1 = 1
K2 = 2
KY = -0.5  # Simple feedback gain for vy commands
# A - BK - LC = [-4  1]
#               [-5 -2]
L1 = 4
L2 = 4

# Stabilization thresholds
X_THRESH = 0.1  # meters
Y_THRESH = 0.05
Z_THRESH = 0.05


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


if __name__ == "__main__":
    print("Starting script...")
    # For tracking flight time
    start_time = time.time()

    # Make a pilot object and take off
    print("Drone taking off")
    pilot = tx.Pilot()
    pilot.takeoff()

    # We're assuming that the drone starts facing the gate, so we don't need to rotate to search
    gate_found = False
    stabilized = False
    while not gate_found and not stabilized:
        # Log sensor readings
        readings = pilot.get_sensor_readings()

        # Current height
        if readings.height > MAX_HEIGHT:
            print("WARNING: Flew too high! Landing for safety...")
            pilot.land()
            break

        # Search for tags
        img = pilot.get_camera_frame(visualize=False)
        tags = pilot.detect_tags(img, visualize=True)
        if tags:
            # Stop only when all tags are in FOV
            if len(tags) == GATE_NUM_TAGS:
                print("Gate found in field of view! Detected {} AprilTags".format(len(tags)))
                gate_found = True
                # Stop increasing altitude
                pilot.send_control(np.zeros(3), 0.0)
        else:
            # Increase altitude to keep searching
            print("Increasing altitude...")
            z_vel = -1 * KY * DELTA_POS
            z_vel = np.clip(z_vel, -1 * MAX_VEL_MAG, MAX_VEL_MAG)
            xyz_velocity = np.array([0.0, 0.0, z_vel])
            pilot.send_control(xyz_velocity, 0.0)  # no yaw
            time.sleep(1 / VEL_CONTROL_RATE)
            print("Sending velocity command: ", xyz_velocity)

        # Land if exceeded max flight time
        if (time.time() - start_time) > MAX_FLIGHT_TIME:
            print("Exceeded max flight time of {} without detecting AprilTag gate.".format(MAX_FLIGHT_TIME))
            print("Landing...")
            pilot.land()
            break

        # Collection containers for sensor readings
        positions = []
        controls_altitude = []
        aprilTag_lost_cnt = 0
        # Initialize observer state
        vz_hat = 0.0
        z_hat = 0.0
        vz = 0.0
        while not stabilized and gate_found:
            print("Stabilizing at gate center...")
            # Log sensor readings
            readings = pilot.get_sensor_readings()

            # Detect new tag positions
            img = pilot.get_camera_frame(visualize=False)
            tags = pilot.detect_tags(img, visualize=True)
            print("Detected {} AprilTags in FOV; expected {}".format(len(tags), GATE_NUM_TAGS))

            # Tends to lose altitude just from drifting so it might lose tags
            if len(tags) < (GATE_NUM_TAGS - 2):
                print("Waiting to re-detect all tags...")
                aprilTag_lost_cnt += 1
                if aprilTag_lost_cnt > 10:
                    print("Lost gate, setting gate_found to False")
                    gate_found = False
                continue

            position = get_pose_gate_center(pilot, tags)
            print("Pose w.r.t. gate center = ", position)
            positions.append(position)

            # Check stabilization thresholds, then land the drone if within bounds
            x_diff = position[0] - X_REF
            y_diff = position[1]
            z_diff = position[2]

            if abs(x_diff) <= X_THRESH and abs(y_diff) <= Y_THRESH and abs(z_diff) <= Z_THRESH:
                stabilized = True
                print("Stabilized at gate center!")
                pilot.land()
                cv2.imwrite("GateVisible.png", img)
                break

            else:
                # Stabilize at the desired point

                # Simple feedback control
                x_vel = KY * x_diff
                y_vel = KY * y_diff

                # Full-state feedback
                # az = -k1*(z_hat - z_ref) - k2*v_z_hat
                # vz[t] = vz[t-1] + az[t]*dt
                z = position[2]
                dt = 1.0 / VEL_CONTROL_RATE
                print("\tvz_hat = ", vz_hat)
                az = -K1 * z_hat - (K2 * vz_hat)
                print("\taz = ", az)
                z_hat_dot = L1*z - L1*z_hat + vz_hat
                vz_hat_dot = L2*z - (K1 + L2)*z_hat - (K2 * vz_hat)
                vz = vz + az * dt

                # Propagate states forward
                z_hat += dt * z_hat_dot
                vz_hat += dt * vz_hat_dot
                print("Computed state = ({}, {}) (m, m/s)".format(z_hat, vz_hat))

                # Control values
                z_vel = vz
                # Make sure to not go over control command bounds
                x_vel = np.clip(x_vel, -1 * MAX_VEL_MAG, MAX_VEL_MAG)
                y_vel = np.clip(y_vel, -1 * MAX_VEL_MAG, MAX_VEL_MAG)
                z_vel = np.clip(z_vel, -1 * MAX_VEL_MAG, MAX_VEL_MAG)

                xyz_velocity = np.array([x_vel, y_vel, z_vel])
                print("Sending stabilization control:")
                print("Velocity: ", xyz_velocity)
                pilot.send_control(xyz_velocity, 0.0)  # no yaw
                controls_altitude.append(xyz_velocity)
                time.sleep(1 / VEL_CONTROL_RATE)
    

    # Plot logged data
    positions = np.array(positions)
    controls_altitude = np.array(controls_altitude)

    np.save("position.npy", positions)
    np.save("controls_altitude.npy", controls_altitude)
    # Positions will have 1 more data point than controls
    np.save("gate_pass_data.npy", np.hstack((controls, positions[1:])))

    # Plot data
    plt.figure()
    plt.plot(positions[:, 0], label="x")
    plt.plot(positions[:, 1], label="y")
    plt.plot(positions[:, 2], label="z")
    plt.ylabel("Relative Position [m]")
    plt.title("Relative States for Gate Pass")
    plt.legend()
    plt.savefig("gate_pass_states.png")

    plt.figure()
    plt.plot(v_xyz[:, 0], label="vx")
    plt.plot(v_xyz[:, 1], label="vy")
    plt.plot(v_xyz[:, 2], label="vz")
    plt.ylabel("Velocity Command [m/s]")
    plt.title("Velocity Commands for Gate Pass")
    plt.legend()
    plt.savefig("gate_pass_commands.png")

