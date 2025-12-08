# core.py
import rclpy
import math

from auto_cube_delivery.modules.navigator import Navigator
from auto_cube_delivery.modules.database import Database

def core_process():
    print("Starting Core Process...")

    # ---- Hyper Parameters ----------------- #
    # map frame
    # x, y, yaw
    landmark = [(1.0, 0.0, 0.0),    # left
                (1.19, 3.24, -87.9),    # middle
                (3.0, 0.0, 0.0)]    # right

    # movable range for initial localization
    cov_threshold = 0.05
    move_range_x = (-0.05, 0.05)
    move_range_y = (-0.03, 0.03)

    use_zone_marker = True
    task_prompt = None
    # --------------------------------------- #

    # Init Navigator
    # localize with threshold covariance
    navigator = Navigator(cov_threshold=cov_threshold, move_range_x=move_range_x, move_range_y=move_range_y)

    # Init Database
    # Add landmark coordinate
    database = Database(left = landmark[0], middle=landmark[1], right=landmark[2])

    # Init Gemini
    # gemini = Gemini()

    # Visiting Order
    landmark_dist = {'left': -1.0, 'middle': -1.0, 'right':-1.0}
    start_point_curr = navigator.start_point
    assert  start_point_curr is not None

    for direction in landmark_dist:
        landmark_dist[direction] = math.dist(database.landmark[direction][0:2], start_point_curr[0:2])
    landmark_visit_order = sorted(landmark_dist, key=landmark_dist.get)

    # Visit from the closest
    for direction in landmark_visit_order:
        navigation_is_done = navigator.set_goal(database.landmark[direction])
        if navigation_is_done:
            print(f"Arrived at Landmark: {direction}")

            # Move Arm to Marker Reading Position

            # Read Image

            # if use_zone_marker:
                # using aruco marker for zone detection

                # Read Zone marker using marker detection code
                # zone_id = marker_detector.detect_id(img)
                # - what if the cube marker is also detected?

                # Fill database
                # database.fill_zone_id(direction, zone_id)

            # else:
                # Not using marker for zone detection
                # Use gemini

                # Read Zone number using gemini
                # zone_num = gemini.ask_zone_num(img)

                # Fill database
                # database.fill_zone_num(direction, zone_num)

            # Read Cube color using Gemini
            # use gemini
            # - cube_color = gemini.ask_cube_color(img)
            #
            # Or use yolo, cut the cube patch, then ask gemini
            # Or use yolo, cut the cube patch, then use color distance metric
            # within (red, blue, green) colors
            #
            # - what if no cube? Empty zone?

            # Fill database
            # database.fill_cube_info(cube_color, direction)

        else:
            print(f"Failed to go to Landmark: {direction}")
            assert False

    navigator.move_to_start()

    # Ask Gemini for Taks action sequence
        # task_seq = gemini.ask_task(task_prompt)
    # Assert type=list, all elements are in action options
    # Fill database with action sequence
        # database.fill_task_seq(task_seq)
    # For loop for actual manipulation
    # for each pick-place pair
        # move to landmark
        # (approach) -> neglect for now
        # pick
        # fill database with grasping anlge

        # move to landmark
        # (approach) -> neglect for now
        # place (based on previouse grasping angle)

    # return to starting point after all tasks are done


if __name__ == '__main__':
    try:
        rclpy.init()
        core_process()
    except KeyboardInterrupt:
        print("Force Shutdown")
    finally:
        rclpy.shutdown()
