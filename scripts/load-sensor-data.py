import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = np.load("sensor_data.npy", allow_pickle=True).item()

    acceleration = data["acceleration"]  # 3x1, m/s^2
    velocity = data["velocity"]  # 3x1, m/s
    attitude = data["attitude"]  # 3x1, deg
    height_sonar = data["height"]  # 1x1, m
    height_baro = data["baro"]  # 1x1, m

    time = np.arange(0, 5, 0.1) # sampled at 10 Hz

    plt.plot(time, acceleration, label=["x", "y", "z"])
    plt.xlabel("Time (s)")  
    plt.ylabel("Acceleration (m/s^2)")
    plt.legend()
    plt.savefig("data-acceleration.png")
    plt.close()

    plt.plot(time, velocity, label=["x", "y", "z"])
    plt.xlabel("Time (s)")  
    plt.ylabel("Velocity (m/s)")
    plt.legend()
    plt.savefig("data-velocity.png")
    plt.close()

    plt.plot(time, attitude, label=["x", "y", "z"])
    plt.xlabel("Time (s)")  
    plt.ylabel("Attitude (deg)")
    plt.legend()
    plt.savefig("data-attitude.png")
    plt.close()

    plt.plot(time, height_sonar)
    plt.ylabel("Height - Sonar (m)")
    plt.savefig("data-sonar.png")
    plt.close()

    plt.plot(time, height_baro)
    plt.ylabel("Height - Baro (m)")
    plt.savefig("data-baro.png")
    plt.close()
