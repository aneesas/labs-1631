import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = np.load(sensor_data)

    acceleration = data["acceleration"]  # 3x1, m/s^2
    velocity = data["velocity"]  # 3x1, m/s
    attitude = data["attitude"]  # 3x1, deg
    height_sonar = data["h"]  # 1x1, m
    height_baro = data["baro"]  # 1x1, m

    # TODO might have to do accel/vel/attitude as 3 separate plot calls
    plt.plot(acceleration)
    plt.xlabel("Time (s)")  # TODO actually I think this will be sample #, not time
    plt.ylabel("Acceleration (m/s^2)")
    plt.legend()
    plt.savefig("data-acceleration.png")
    plt.close()

    plt.plot(velocity)
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.savefig("data-velocity.png")
    plt.close()

    plt.plot(attitude)
    plt.ylabel("Attitude (deg)")
    plt.legend()
    plt.savefig("data-attitude.png")
    plt.close()

    plt.plot(height_sonar)
    plt.ylabel("Height - Sonar (m)")
    plt.legend()
    plt.savefig("data-sonar.png")
    plt.close()

    plt.plot(height_baro)
    plt.ylabel("Height - Baro (m)")
    plt.legend()
    plt.savefig("data-baro.png")
    plt.close()
