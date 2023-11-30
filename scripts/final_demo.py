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
VEL_CONTROL_RATE = 5.0  # Hz
MAX_FLIGHT_TIME = 100  # seconds
GATE_NUM_TAGS = 8  # defines how many AprilTags make up a full gate
YAW_RATE = 25.0  # deg/sec; keep this within [-100, 100]
YAW_CONTROL_RATE = 100.0  # Hz

# Control law values
X_REF = -2.0  # meters; for stabilizing drone relative to AprilTag gate
K1 = 1
K2 = 2
KY = -0.5  # Simple feedback gain for vy commands

# Stabilization thresholds
X_THRESH = 0.1  # meters
Y_THRESH = 0.15
Z_THRESH = 0.5


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
    stabilized = False
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
            # TODO we could slow down the velocity here
            # Stop only when all tags are in FOV
            if len(tags) == GATE_NUM_TAGS:
                print("Gate found in field of view! Detected {} AprilTags".format(len(tags)))
                gate_found = True
                # Stop increasing altitude
                pilot.send_control(np.zeros(3), 0.0)
        else:
            # Increase altitude to keep searching
            print("Increasing altitude...")
            # az = k1*(z - z_ref) + k2*v_z_hat
            # vz[t] = vz[t-1] + az[t]*dt
            # vz = K1 * (z - Z_REF) TODO
            vz = -1 * KY * DELTA_POS
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

            position = get_pose_gate_center(pilot, tags)
            print("Pose w.r.t. gate center = ", position)
            positions.append(position)

            # Check stabilization thresholds, then land the drone if within bounds
            x_diff = position[0] - X_REF
            y_diff = -position[1]
            z_diff = position[2]

            if abs(x_diff) <= X_THRESH and abs(y_diff) <= Y_THRESH and abs(z_diff) <= Z_THRESH:
                stabilized = True
                print("Stabilized at gate center!")
                cv2.imwrite("GateVisible.png", img)
                break

            else:
                # Stabilize at the desired point
                # TODO replace with actual controllers
                x_vel = KY * x_diff
                y_vel = KY * y_diff
                z_vel = KY * z_diff
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


    print("Flying through gate...")
    x_dist = 4.0  # m
    x_vel = 0.3  # m/s
    fly_open_loop(pilot, np.array([x_vel, 0.0, 0.0]), 6.5, VEL_CONTROL_RATE)

    # Wait for the drone to completely stop flying forward, because it takes a second
    time.sleep(1.0)

    # Turn around after passing through gate
    deg_turned = 0.0
    readings = pilot.get_sensor_readings()
    yaw_prev = readings.attitude[2]
    print("Initial yaw angle = ", yaw_prev)
    while (deg_turned < 180):
        readings = pilot.get_sensor_readings()
        yaw_angle = readings.attitude[2]
        deg_turned += abs(yaw_angle - yaw_prev)
        print("Current angle = {} deg; total deg turned = {}".format(yaw_angle, deg_turned))
        yaw_prev = yaw_angle
        pilot.send_control(np.array([0.0, 0.0, 0.0]), -YAW_RATE)
        time.sleep(1.0/YAW_CONTROL_RATE)
        img = pilot.get_camera_frame(visualize=False)
        tags = pilot.detect_tags(img, visualize=True)
        if tags and len(tags) == (GATE_NUM_TAGS - 2):
            print("Found gate while turning! Stopping turn...")
            pilot.send_control(np.array([0.0, 0.0, 0.0]), 0.0)
    pilot.send_control(np.array([0.0, 0.0, 0.0]), 0.0)
    print("Completed turn.")

    gate_found = False
    search_time = 0.0
    while not gate_found and search_time < 1.5:
        # Search for tags
        img = pilot.get_camera_frame(visualize=False)
        tags = pilot.detect_tags(img, visualize=True)

        if tags and len(tags) == GATE_NUM_TAGS:
            print("Gate found in field of view! Detected {} AprilTags".format(len(tags)))
            pilot.send_control(np.array([0.0, 0.0, 0.0]), 0.0)
            gate_found = True
            cv2.imwrite("GateVisibleAgain.png", img)
            pilot.land()
        else:
            print("Only detected {} AprilTags; expected {}".format(len(tags), GATE_NUM_TAGS))
            # Fly backwards SLOWLY
            print("Backing drone up...")
            pilot.send_control(np.array([-0.1, 0.0, 0.0]), 0.0)
            time.sleep(1.0 / VEL_CONTROL_RATE)
            search_time += 1.0/VEL_CONTROL_RATE
    print("Landing...")
    pilot.land()

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
