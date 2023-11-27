import time
import matplotlib.pyplot as plt
import numpy as np
import cv2

import tellox as tx

# Constants
YAW_RATE = 40.0  # deg/sec; keep this within [-100, 100]
MAX_HEIGHT = 2  # meters; depends on room
MAX_VEL_MAG = 1.0  # m/s
YAW_CONTROL_RATE = 100.0  # Hz
VEL_CONTROL_RATE = 5.0  # Hz
MAX_FLIGHT_TIME = 300  # seconds

# Control law values
Z_REF = 2.5  # meters; for initial search
K1 = -1  # gain for initial search

X_REF = -1  # meters; for stabilizing drone relative to AprilTag
YAW_REF_DEG = 0  # degrees; for stabilizing drone relative to AprilTag
KXYZ = -0.5  # gain for stabilizing drone

# Stabilize Thresholds
X_THRESH = 0.1  # meters
Y_THRESH = 0.1
Z_THRESH = 0.1
YAW_THRESH_DEG = 5  # degrees

if __name__ == "__main__":
    print("Starting script...")
    # For tracking flight time
    start_time = time.time()

    # Collection containers for sensor readings
    yaw_angles = []
    altitudes = []
    times = []

    # Make a pilot object and take off
    print("Drone taking off")
    pilot = tx.Pilot()
    pilot.takeoff()

    # Rotate 360 degrees clockwise at current altitude, searching for AprilTag
    apriltag_found = False
    stabilized = False
    prev_yaw_angle = pilot.get_sensor_readings().attitude[2]
    deg_turned = 0
    img_num = 0
    while not apriltag_found:
        # Log sensor readings
        readings = pilot.get_sensor_readings()
        yaw_angles.append(readings.attitude[2])
        altitudes.append(readings.height)
        times.append(readings.flight_time)

        # Current height
        z = altitudes[-1]
        if z > MAX_HEIGHT:
            print("WARNING: Flew too high! Landing for safety...")
            pilot.land()
            break

        # Look for AprilTag
        img = pilot.get_camera_frame(visualize=False)
        tags = pilot.detect_tags(img, visualize=True)
        if tags:
            print("Detected AprilTag!")
            apriltag_found = True

            controls = []
            positions = []
            angles = []
            while not stabilized:
                pilot.send_control(np.array([0.0, 0.0, 0.0]), 0.0)
                position , _, euler_angles = pilot.get_drone_pose(tags[0])  # returns angles in radians!!
                euler_angles = euler_angles * 180. / np.pi  # convert to degrees
                print('Position:', position)
                print('RPY angles (deg):', euler_angles)
                positions.append(position)
                angles.append(euler_angles)

                # Check stabilization thresholds, then land the drone if within bounds
                # Drone should be at the center of the AprilTag, X_REF meters away
                x_diff = position[0] - X_REF
                y_diff = position[1]  # relative y-z position should be 0 (reference position of 0)
                z_diff = position[2]
                yaw_diff = euler_angles[2] - YAW_REF_DEG
                if (abs(x_diff) <= X_THRESH and abs(y_diff) <= Y_THRESH and abs(z_diff) <= Z_THRESH and abs(yaw_diff) <= YAW_THRESH_DEG):
                    stabilized = True
                    pilot.land()
                    cv2.putText(img, "stabilized",
                        (int(tags[0].center[0]), int(tags[0].center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.imwrite("AprilTagCentered.png", img)

                # Stabilize at the desired point
                x_vel = KXYZ * x_diff
                y_vel = KXYZ * y_diff
                z_vel = KXYZ * z_diff
                yaw_vel = KXYZ * yaw_diff
                controls.append((x_vel, y_vel, z_vel, yaw_vel))

                # Make sure to not go over control command bounds
                x_vel = np.clip(x_vel, -1 * MAX_VEL_MAG, MAX_VEL_MAG)
                y_vel = np.clip(y_vel, -1 * MAX_VEL_MAG, MAX_VEL_MAG)
                z_vel = np.clip(z_vel, -1 * MAX_VEL_MAG, MAX_VEL_MAG)
                if abs(yaw_vel) > 100:
                    print("WARNING: Massive yaw rate command! Doesn't make sense!")
                    np.clip(yaw_vel, -100, 100)

                xyz_velocity = np.array([x_vel, y_vel, z_vel])
                print("Sending stabilization control:")
                print("Velocity: ", xyz_velocity)
                print("Yaw rate: ", yaw_vel)
                pilot.send_control(xyz_velocity, yaw_vel)
                # Control inputs can be pretty aggressive; let it run for a short amount of time before stopping
                time.sleep(1 / VEL_CONTROL_RATE)
                pilot.send_control(np.array([0.0, 0.0, 0.0]), 0.0)

                # Detect new tag position for next iteration
                img = pilot.get_camera_frame(visualize=False)
                tags = pilot.detect_tags(img, visualize=True)

                # Drone sometimes drops detection of AprilTag despite it being in view
                # If this happens, try a couple times to re-detect tag
                retries = 3
                tags_counter = 0
                if not tags:
                    while not tags and tags_counter <= retries:
                        tags = pilot.detect_tags(img, visualize=True)
                        tags_counter += 1
                    # If drone can't see tag after a few tries, it has probably drifted out of the FOV
                    # Return to the outer loop to redetect
                    if tags_counter > retries:
                        apriltag_found = False
                        break
                
        else:
            # Check if we've done a full 360 search
            deg_turned += np.abs(yaw_angles[-1] - prev_yaw_angle)
            prev_yaw_angle = yaw_angles[-1]
            if deg_turned > 180:
                print("Increasing altitude...")
                vz = K1 * (z - Z_REF)
                vz = np.clip(vz, -1 * MAX_VEL_MAG, MAX_VEL_MAG)
                xyz_velocity = np.array([0.0, 0.0, vz])
                pilot.send_control(xyz_velocity, YAW_RATE)
                time.sleep(1 / VEL_CONTROL_RATE)
                deg_turned = 0
            else:
                xyz_velocity = np.array([0.0, 0.0, 0.0])
                pilot.send_control(xyz_velocity, YAW_RATE)
                time.sleep(1 / YAW_CONTROL_RATE)
        # Save imagery
        cv2.imwrite("./images/img" + str(img_num) + ".jpg", img)
        img_num += 1

        # Land if exceeded max flight time
        if (time.time() - start_time) > MAX_FLIGHT_TIME:
            print("Exceeded max flight time of {} without detecting AprilTag.".format(MAX_FLIGHT_TIME))
            print("Landing...")
            pilot.land()
            break

    # For debugging/analysis:
    # Plot logged data
    positions = np.array(positions)
    angles = np.array(angles)
    controls = np.array(controls)

    plt.figure()
    plt.plot(angles[:, 0], "r-", label="roll")
    plt.plot(angles[:, 1], "b-", label="pitch")
    plt.plot(angles[:, 2], "g-", label="yaw")
    plt.ylabel("Euler angle (deg)")
    plt.legend()
    plt.savefig("euler_angles.png")

    plt.figure()
    plt.plot(angles[:, 2])
    plt.ylabel("Stabilization yaw angle (deg)")
    plt.savefig("EA_yaw.png")

    plt.figure()
    plt.plot(controls[:, 0], label="x_vel")
    plt.plot(controls[:, 1], label="y_vel")
    plt.plot(controls[:, 2], label="z_vel")
    plt.ylabel("Velocity control (m/s)")
    plt.legend()
    plt.savefig("vel_controls.png")

    plt.figure()
    plt.plot(controls[:, 3])
    plt.ylabel("Yaw control (deg/s)")
    plt.savefig("yaw_controls.png")

    plt.figure()
    plt.plot(positions[:, 0], label="x")
    plt.plot(positions[:, 1], label="y")
    plt.plot(positions[:, 2], label="z")
    plt.title("Relative position of drone in AprilTag frame")
    plt.ylabel("Rel. position (m)")
    plt.legend()
    plt.savefig("position_xyz.png")

    data = np.array((positions, angles), dtype=float)
    np.save("relative_location.npy", data)
