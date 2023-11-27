import time
import matplotlib.pyplot as plt
import numpy as np
import cv2

import tellox as tx

# Constants
YAW_RATE = 40.0  # deg/sec; keep this within [-100, 100]
MAX_HEIGHT = 2  # meters; depends on room
MAX_VZ_MAG = 1.0  # m/s
YAW_CONTROL_RATE = 100.0  # Hz
VEL_CONTROL_RATE = 5.0  # Hz
MAX_FLIGHT_TIME = 80  # seconds

# Control law values
Z_REF = 2.5
K1 = -1

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
            print("Detected AprilTag! Landing...")
            apriltag_found = True
            pilot.land()
            cv2.putText(img, "detected",
                (int(tags[0].center[0]), int(tags[0].center[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imwrite("AprilTagVisible.png", img)
        else:
            # Check if we've done a full 360 search
            deg_turned += np.abs(yaw_angles[-1] - prev_yaw_angle)
            prev_yaw_angle = yaw_angles[-1]
            if deg_turned > 360:
                print("Completed 360 degree search, increasing altitude...")
                vz = K1*(z - Z_REF)
                vz_control.append(vz)
                vz = np.clip(vz, -1 * MAX_VZ_MAG, MAX_VZ_MAG)
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
