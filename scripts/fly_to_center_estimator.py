import time
import matplotlib.pyplot as plt
import numpy as np
import cv2

import tellox as tx

from utils import *

# Constants/environment definition
MAX_HEIGHT = 5  # meters; depends on room, used for safety checks
MAX_VEL_MAG = 0.75  # m/s; for safety
DELTA_POS = 0.5  # increments for sending velocity commands
Z_REF = 2.0  # target height for finding the AprilTags
VEL_CONTROL_RATE = 5.0  # Hz
MAX_FLIGHT_TIME = 100  # seconds
GATE_NUM_TAGS = 8  # defines how many AprilTags make up a full gate

# Control law values
X_REF = -2.0  # meters; for stabilizing drone relative to AprilTag gate
K1 = 1
K2 = 2
KY = -0.5  # Simple feedback gain for vy commands
# A-LC = [-4 1]
#        [-4 0]
A_OBSERVER = np.array([[-4, 1], [-4, 0]])

# Stabilization thresholds
X_THRESH = 0.1  # meters
Y_THRESH = 0.1
Z_THRESH = 0.1


def compute_vz(x_prev, vz_prev, dt, dz):
    # x = [z_hat, vz_hat]
    # az = k1*(z - z_ref) + k2*v_z_hat
    # vz[t] = vz[t-1] + az[t]*dt
    x = x_prev + dt * np.matmul(A_OBSERVER, x_prev)
    vz_hat = x[-1]  # from state definition
    print("\tvz_hat = ", vz_hat)
    az = K1 * dz + K2 * vz_hat
    print("\taz = ", az)
    vz = vz_prev + az * dt
    return(x, vz)


if __name__ == "__main__":
    print("Starting script...")
    # For tracking flight time
    start_time = time.time()

    # Make a pilot object and take off
    print("Drone taking off")
    pilot = tx.Pilot()
    # pilot.takeoff()

    # Collection containers for sensor readings
    altitudes = []
    times = []
    controls_altitude = []

    # Get initial sensor readings
    readings = pilot.get_sensor_readings()
    altitudes.append(readings.height)
    times.append(readings.flight_time)

    # We're assuming that the drone starts facing the gate, so we don't need to rotate to search
    gate_found = False
    stabilized = False
    # Initialize observer state to [z_hat = altitude measurement, vz_hat = 0]
    state_prev = np.array([-DELTA_POS, 0.0])
    print("Initial state = {} (m, m/s)".format(state_prev))
    vz_prev = 0.0
    while not gate_found and not stabilized:
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
            # Stop only when all tags are in FOV
            if len(tags) == GATE_NUM_TAGS:
                print("Gate found in field of view! Detected {} AprilTags".format(len(tags)))
                gate_found = True
                # Stop increasing altitude
                pilot.send_control(np.zeros(3), 0.0)
        else:
            # Increase altitude to keep searching
            print("Increasing altitude...")
            state, vz = compute_vz(state_prev, vz_prev, 1/VEL_CONTROL_RATE, altitudes[-1] - Z_REF)
            print("Computed state = {} (m, m/s)".format(state))
            # Save actual computed values
            controls_altitude.append(vz)
            # vz = np.clip(vz, -1 * MAX_VEL_MAG, MAX_VEL_MAG)
            xyz_velocity = np.array([0.0, 0.0, vz])  # TODO -vz? or vz?
            print("Sending velocity command: ", xyz_velocity)
            # pilot.send_control(xyz_velocity, 0.0)  # no yaw
            time.sleep(1 / VEL_CONTROL_RATE)
            state_prev = state
            vz_prev = vz
            print("state_prev = ", state_prev)
            print("vz_prev = ", vz_prev)

        # Land if exceeded max flight time
        if (time.time() - start_time) > MAX_FLIGHT_TIME:
            print("Exceeded max flight time of {} without detecting AprilTag gate.".format(MAX_FLIGHT_TIME))
            print("Landing...")
            pilot.land()
            break

        controls_gate = []  # to store velocity control commands
        positions = []
        aprilTag_lost_cnt = 0
        while not stabilized and gate_found:
            print("Stabilizing at gate center...")
            # Log sensor readings
            readings = pilot.get_sensor_readings()
            altitudes.append(readings.height)
            times.append(readings.flight_time)

            # Detect new tag positions
            img = pilot.get_camera_frame(visualize=False)
            tags = pilot.detect_tags(img, visualize=True)
            print("Detected {} AprilTags in FOV; expected {}".format(len(tags), GATE_NUM_TAGS))
            # TODO just keep looping if it doesn't work?
            if len(tags) < (GATE_NUM_TAGS - 2):
                print("Waiting to re-detect all tags...")
                aprilTag_lost_cnt += 1
                if aprilTag_lost_cnt > 10:
                    print("Lost gate, setting gate_found to False")
                    gate_found = False
                continue

            # Use the top-left gate tag as the reference point
            # reference_tag_idx = get_top_left_idx(tags)
            # gate_center_dist_yz = get_gate_center(pilot, tags, reference_tag_idx)
            # print("Gate center wrt tag ID {} = {}".format(reference_tag_idx, gate_center_dist_yz))

            # position , _, _ = pilot.get_drone_pose(tags[reference_tag_idx])
            # print("Current position wrt reference tag:", position)
            # positions.append(position)
            position = get_pose_gate_center(pilot, tags)
            print("Pose w.r.t. gate center = ", position)
            positions.append(position)

            # Check stabilization thresholds, then land the drone if within bounds
            x_diff = position[0] - X_REF
            y_diff = -position[1]
            z_diff = position[2]

            if abs(x_diff) <= X_THRESH and abs(y_diff) <= Y_THRESH and abs(z_diff) <= Z_THRESH:
                stabilized = True
                # TODO land just for now
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
                state, z_vel = compute_vz(state_prev, vz_prev, 1/VEL_CONTROL_RATE, z_diff)
                print("Computed state = {} (m, m/s)".format(state))
                controls_gate.append((x_vel, y_vel, z_vel))
                state_prev = state
                vz_prev = z_vel

                # Make sure to not go over control command bounds
                x_vel = np.clip(x_vel, -1 * MAX_VEL_MAG, MAX_VEL_MAG)
                y_vel = np.clip(y_vel, -1 * MAX_VEL_MAG, MAX_VEL_MAG)
                z_vel = np.clip(z_vel, -1 * MAX_VEL_MAG, MAX_VEL_MAG)

                xyz_velocity = np.array([x_vel, y_vel, -z_vel])
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

    # plt.figure()
    # plt.plot(controls_altitude[:, 0], label="x_vel")
    # plt.plot(controls_altitude[:, 1], label="y_vel")
    # plt.plot(controls_altitude[:, 2], label="z_vel")
    # plt.ylabel("Velocity control (m/s)")
    # plt.title("Velocity control to detect gate")
    # plt.legend()
    # plt.savefig("controls_altitude.png")

    # plt.figure()
    # plt.plot(controls_gate[:, 0], label="x_vel")
    # plt.plot(controls_gate[:, 1], label="y_vel")
    # plt.plot(controls_gate[:, 2], label="z_vel")
    # plt.ylabel("Velocity control (m/s)")
    # plt.title("Velocity control to center")
    # plt.legend()
    # plt.savefig("controls_gate.png")

    # plt.figure()
    # plt.plot(positions[:, 0], label="x")
    # plt.plot(positions[:, 1], label="y")
    # plt.plot(positions[:, 2], label="z")
    # plt.title("Relative position of drone in top-left AprilTag frame")
    # plt.ylabel("Rel. position (m)")
    # plt.legend()
    # plt.savefig("position_xyz.png")
