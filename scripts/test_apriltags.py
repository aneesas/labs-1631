import time
import numpy as np

import tellox as tx

if __name__ == "__main__":
    print("Starting script...")

    pilot = tx.Pilot()

    gate_found = False
    while not gate_found:
        img = pilot.get_camera_frame(visualize=False)
        tags = pilot.detect_tags(img, visualize=True)
        if tags:
            print("Detected {} AprilTags in image!".format(len(tags)))
            print("tag.pose_t = ", tags[0].pose_t.reshape(3,))
            print("pose in global frame = ", pilot.get_drone_pose(tags[0]))