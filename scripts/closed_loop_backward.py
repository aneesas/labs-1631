import time
import matplotlib.pyplot as plt
import numpy as np
import cv2

import tellox as tx

from utils import *

# Constants
MAX_HEIGHT = 2.5  # meters; depends on room, used for safety checks
MAX_VEL_MAG = 0.5  # m/s; for safety
VEL_CONTROL_RATE = 5.0  # Hz
OPEN_LOOP_VEL = 0.25  # m/s
MAX_FLIGHT_TIME = 60  # seconds
GATE_NUM_TAGS = 4  # defines how many AprilTags make up a full gate

# Control law values
X_OPEN_LOOP = 1.5  # 5 ft to meters
X_REF_FINAL = -3.0  # 10 ft to meters
Z_REF = 1.5  # meters; TODO (pick roughly the center of the gate)
K = -0.5  # Simple feedback gain

# Stabilization thresholds
X_THRESH = 0.1  # meters
Z_THRESH = 0.4  # noisier?

if __name__ == "__main__":
    print("Starting script...")

    # Collection containers for sensor readings
    altitudes = []
    controls = []
    positions = []

    # Make a pilot object and take off
    print("Drone taking off")
    pilot = tx.Pilot()
    pilot.takeoff()

    # For tracking flight time
    start_time = time.time()

    # Assume AprilTags won't be in FOV because of how close we're starting to the gate
    # So instead, we fly to a fixed height
    stabilized = False
    while not stabilized:
        # Log sensor readings
        readings = pilot.get_sensor_readings()
        altitudes.append(readings.height)

        # Check current height
        z = altitudes[-1]
        if z > MAX_HEIGHT:
            print("WARNING: Flew too high! Landing for safety...")
            pilot.land()
            break

        # Land if exceeded max flight time
        if (time.time() - start_time) > MAX_FLIGHT_TIME:
            print("Exceeded max flight time of {} without detecting AprilTag gate.".format(MAX_FLIGHT_TIME))
            print("Landing...")
            pilot.land()
            break

        # Check if we've reached reference height
        z_diff = z - Z_REF
        if abs(z_diff) <= Z_THRESH:
            print("Reached reference height of {} meters.".format(Z_REF))
            stabilized = True
        else:
            # Bring drone to reference height
            vz = K * z_diff
            vz = np.clip(vz, -1 * MAX_VEL_MAG, MAX_VEL_MAG)
            print("Increasing altitude to {} meters...".format(Z_REF))
            xyz_velocity = np.array([0.0, 0.0, vz])
            controls.append(xyz_velocity)
            pilot.send_control(xyz_velocity, 0.0)
            time.sleep(1.0 / VEL_CONTROL_RATE)
            pilot.send_control(np.array[0.0, 0.0, 0.0], 0.0)


    # Open-loop flight to X_REF distance
    # We're assuming the drone started at a known distance away from the wall/gate
    print("Flying open-loop to move backwards {} meters in x...")
    reference_reached = False
    gate_found = False
    time_elapsed = 0.0
    total_time = X_OPEN_LOOP / OPEN_LOOP_VEL  # seconds
    dt = 1.0 / VEL_CONTROL_RATE
    z0 = pilot.get_sensor_readings().height  # initial z to maintain

    while not reference_reached and not gate_found:
        # Log sensor readings
        readings = pilot.get_sensor_readings()
        altitudes.append(readings.height)
        z = altitudes[-1]

        # Look for AprilTags
        img = pilot.get_camera_frame(visualize=False)
        tags = pilot.detect_tags(img, visualize=True)
        if tags:
            print("Detected {} AprilTags in FOV; looking for {} for gate".format(len(tags), GATE_NUM_TAGS))
            if not gate_found and len(tags) == GATE_NUM_TAGS:
                print("Found gate!")
                gate_found = True
                cv2.imwrite("Apriltags_again_visible.png", img)
                # Just in case
                pilot.send_control([0.0, 0.0, 0.0], 0.0)

        # Keep z stable
        z_diff = z - z0
        if abs(z_diff) > Z_THRESH:
            vz = K * z_diff
            vz = np.clip(vz, -1 * MAX_VEL_MAG, MAX_VEL_MAG)
        else:
            vz = 0.0

        # Fly for estimated amount of time needed to traverse desired distance
        if not gate_found:
            if time_elapsed < total_time:
                xyz_velocity = np.array([OPEN_LOOP_VEL, 0.0, vz])
                pilot.send_control(xyz_velocity, 0.0)
                controls.append(xyz_velocity)
                time.sleep(dt)
                time_elapsed += dt
            else:
                print("Flew too far without detecting gate; landing")
                pilot.land()
                break
        # With the gate, we can fly closed-loop
        else:
            position = get_pose_gate_center(pilot, tags)
            print("Pose w.r.t. gate center = ", position)
            positions.append(position)
            x_diff = position[0] - X_REF_FINAL
            y_diff = -position[1]
            z_diff = position[2]
            if abs(x_diff) <= X_THRESH:
                print("Reached desired final distance from wall.")
                reference_reached = True
                print("Landing...")
                pilot.land()
            else:
                vx = K * x_diff
                # Only send y/z commands if we drift too much
                # Otherwise motion can be jumpy because controls are imprecise
                vy = K * y_diff if abs(y_diff) > Y_THRESH else 0.0
                vz = K * z_diff if abs(z_diff) > Z_THRESH else 0.0

                vx = np.clip(vz, -1 * MAX_VEL_MAG, MAX_VEL_MAG)
                vy = np.clip(vz, -1 * MAX_VEL_MAG, MAX_VEL_MAG)
                vz = np.clip(vz, -1 * MAX_VEL_MAG, MAX_VEL_MAG)

                xyz_velocity = np.array([vx, vy, vz])
                controls.append(xyz_velocity)
                pilot.send_control(xyz_velocity, 0.0)
                time.sleep(dt)
                pilot.send_control(np.array([0.0, 0.0, 0.0]), 0.0)

    # Plot logged data
    positions = np.array(positions)
    controls = np.array(controls)

    plt.figure()
    plt.plot(controls[:, 0], label="vx")
    plt.plot(controls[:, 1], label="vy")
    plt.plot(controls[:, 2], label="vz")
    plt.ylabel("Velocity control (m/s)")
    plt.title("Velocity control commands")
    plt.legend()
    plt.savefig("closed_loop_backwards_commands.png")

    plt.figure()
    plt.plot(positions[:, 0], label="x")
    plt.plot(positions[:, 1], label="y")
    plt.plot(positions[:, 2], label="z")
    plt.title("Relative position of drone from center of gate")
    plt.ylabel("Rel. position (m)")
    plt.legend()
    plt.savefig("closed_loop_backward_states.png")
