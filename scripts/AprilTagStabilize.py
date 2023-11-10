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
X_REF = -1
Y_REF = 0
Z_REF = 2.5
YAW_REF = 0
K1 = -1
KXYZ = -1

# Stabilize Thresholds
x_thresh = 0.1
y_thresh = 0.1
z_thresh = 0.1
yaw_thresh = 5

if __name__ == "__main__":
    print("Starting script...")
    # For tracking flight time
    start_time = time.time()

    # Collection containers for sensor readings
    yaw_angles = []
    altitudes = []
    times = []
    vz_control = []

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
            while not stabilized:
                print("Detected AprilTag!")
                apriltag_found = True
                pilot.send_control(np.array([0.0, 0.0, 0.0]), 0.0)
                position , _, euler_angles = pilot.get_drone_pose(tags[0])
                print('Position:', position)

                # Check stabilization thresholds, then land the drone
                x_diff = abs(position[0]-X_REF) <= x_thresh
                y_diff = abs(position[1]-Y_REF) <= y_thresh
                z_diff = abs(position[2]-Z_REF) <= z_thresh
                yaw_diff = abs(euler_angles[2]-YAW_REF) <= yaw_thresh
                if (x_diff and y_diff and z_diff and yaw_diff):
                    stabilized = True
                    pilot.land()
                    cv2.putText(img, "detected",
                        (int(tags[0].center[0]), int(tags[0].center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.imwrite("AprilTagVisible.png", img)
                
                # Stabilize at the center of the AprilTag
                x_vel = KXYZ*(position[0] - X_REF)
                x_vel = np.clip(x_vel, -1 * MAX_VEL_MAG, MAX_VEL_MAG)
                y_vel = KXYZ*(position[1] - Y_REF)
                y_vel = np.clip(y_vel, -1 * MAX_VEL_MAG, MAX_VEL_MAG)
                z_vel = KXYZ*(position[2] - 0) 
                z_vel = np.clip(z_vel, -1 * MAX_VEL_MAG, MAX_VEL_MAG)
                yaw_vel = KXYZ*(euler_angles[2] - YAW_REF)
                xyz_velocity = np.array([x_vel, y_vel, z_vel]) 
                pilot.send_control(xyz_velocity, yaw_vel)
                time.sleep(1 / VEL_CONTROL_RATE)
                pilot.send_control(np.array([0.0, 0.0, 0.0]), 0.0)
                img = pilot.get_camera_frame(visualize=False)
                tags = pilot.detect_tags(img, visualize=True)
                tags_counter = 0
                if not tags:
                    while not tags and tags_counter <= 3:
                        tags = pilot.detect_tags(img, visualize=True)
                        tags_counter += 1
                    if tags_counter > 3:
                        apriltag_found = False
                        break
                
        else:
            # Check if we've done a full 360 search
            deg_turned += np.abs(yaw_angles[-1] - prev_yaw_angle)
            prev_yaw_angle = yaw_angles[-1]
            if deg_turned > 360:
                print("Completed 360 degree search, increasing altitude...")
                vz = K1*(z - Z_REF)
                vz_control.append(vz)
                vz = np.clip(vz, -1 * MAX_VEL_MAG, MAX_VEL_MAG)
                xyz_velocity = np.array([0.0, 0.0, vz])
                pilot.send_control(xyz_velocity, YAW_RATE)
                time.sleep(1 / VEL_CONTROL_RATE)
                deg_turned = 0
            else:
                xyz_velocity = np.array([0.0, 0.0, 0.0])
                print("Sending yaw command...")
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
    plt.figure()
    plt.plot(times, yaw_angles, "r-")
    plt.xlabel("Time since takeoff (s)")
    plt.ylabel("Yaw (deg)")
    plt.savefig("yaw.png")

    plt.figure()
    plt.plot(times, altitudes, "r-")
    plt.ylabel("Altitude (m)")
    plt.xlabel("Time since takeoff (s)")
    plt.savefig("h.png")

    fig = plt.figure()
    plt.plot(vz_control)
    plt.ylabel("m/s")
    plt.savefig("vz_control.png")

    data = np.array((altitudes, yaw_angles), dtype=float)
    np.save("data.npy", data)
