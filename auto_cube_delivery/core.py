# core.py
import rclpy

from auto_cube_delivery.modules.navigator import Navigator
from auto_cube_delivery.modules.database import Database

def core_process():
    print("Starting Core Process...")

    # ---- Hyper Parameters ----------------- #
    # map frame
    # x, y, yaw
    landmark = [(0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0)]

    # movable range for initial localization
    cov_threshold = 0.05
    move_range_x = (-0.05, 0.05)
    move_range_y = (-0.03, 0.03)
    # --------------------------------------- #

    navigator = Navigator(cov_threshold=cov_threshold, move_range_x=move_range_x, move_range_y=move_range_y)
    database = Database(left = landmark[0], middle=landmark[1], right=landmark[2])

    landmark_visit_order = ['left', 'middle', 'right']
    for direction in landmark_visit_order:
        navigation_is_done = navigator.set_goal(database.landmark[direction], mode='degrees')
        if navigation_is_done:
            print(f"Arrived at Landmark: {direction}")
        else:
            print(f"Failed to go to Landmark: {direction}")
            assert False

        # add database filling code..

if __name__ == '__main__':
    try:
        rclpy.init()
        core_process()
    except KeyboardInterrupt:
        print("Force Shutdown")
    finally:
        rclpy.shutdown()