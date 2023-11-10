from time import sleep

import numpy as np
import tellox as tx

if __name__ == "__main__":
    pilot = tx.Pilot()

    # Request sensor readings
    readings = []
    flight_time = 5.0  # seconds
    framerate = 10.0  # Hz
    print("Starting measurements!")

    for _ in range(int(flight_time * framerate)):
        readings.append(pilot.get_sensor_readings())
        sleep(1 / framerate)
    print("Done with measurements!")

    # Convert the readings to numpy arrays
    readings_dict = tx.aggregate_sensor_readings(readings)

    # Save to npy file
    np.save("sensor_data", readings_dict)
