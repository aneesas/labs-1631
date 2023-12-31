"""An example of using tellox to control the drone."""
from time import perf_counter, sleep

import matplotlib.pyplot as plt
import numpy as np

import tellox as tx

# Make a pilot object and take off
pilot = tx.Pilot()
pilot.takeoff()

# We'll get the drone to go up and down in a sine wave, and hopefully we
# can see that in the sensor readings
readings = []
controls = []
times = []
flight_time = 10.0  # seconds
framerate = 100.0  # Hz
h = 0.0  # altitude

for t in np.arange(0, flight_time, 1 / framerate):
    times.append(perf_counter())

    # Get the sensor readings (save for later)
    readings.append(pilot.get_sensor_readings())

    # Send a control command to the drone (save for later)
    h = pilot.get_sensor_readings().height
    yaw_rate = 0.0  # rad/s
    if h < 0.9:
        vz = 0.1
    elif h > 1.1:
        vz = -0.1
    else:
        vz = 0
    xyz_velocity = np.array([0.0, 0.0, vz])
    controls.append(xyz_velocity)
    pilot.send_control(xyz_velocity, yaw_rate)

    # Wait for the set control rate
    sleep(1 / framerate)

pilot.land()

# Convert the controls and readings to numpy arrays and plot them
readings_dict = tx.aggregate_sensor_readings(readings)
controls_np = np.array(controls)
plt.plot(
    readings_dict["flight_time"],
    readings_dict["acceleration"][:, 0],
    linestyle="-",
    label="x",
)
plt.plot(
    readings_dict["flight_time"],
    readings_dict["acceleration"][:, 1],
    linestyle="-",
    label="y",
)
plt.plot(
    readings_dict["flight_time"],
    readings_dict["acceleration"][:, 2],
    linestyle="-",
    label="z",
)
plt.plot(
    readings_dict["flight_time"], controls_np[:, 0], linestyle="--", label="vx"
)
plt.plot(
    readings_dict["flight_time"], controls_np[:, 1], linestyle="--", label="vy"
)
plt.plot(
    readings_dict["flight_time"], controls_np[:, 2], linestyle="--", label="vz"
)
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.legend()
plt.savefig("altitude-constant-hover.png")
