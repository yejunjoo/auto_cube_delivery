# core.py
import rclpy
import math
import time


from auto_cube_delivery.modules.navigator import Navigator
from auto_cube_delivery.modules.database import Database

from auto_cube_delivery.modules.plan_generator import create_robot_plan, parse_final_plan
from auto_cube_delivery.modules.zone_cube_analyzer import ZoneCubeAnalyzer

from auto_cube_delivery.modules.grasping_module.grasping import GraspingNode

def core_process():
    print("Starting Core Process...")

    # ---- Hyper Parameters ----------------- #
    # map frame
    # x, y, yaw
    # map_1212
    landmark = [(0.955, -0.805, -56.933),    # left
                (1.720, 0.203, -49.377),    # middle
                (2.538, 1.145, -48.462)]    # right

    # map_1212_2
    landmark = [(0.731, 2.468, -4.0), # left
                (0.766, 1.287, -4.586), # middle
                (0.752, 0.129, -2.723)] # right
    landmark = [(0.731, 2.468, -4.0), # left
                (0.766, 1.287, -4.586), # middle
                (0.752, 0.129, -2.723)] # right


    landmark = [(),
                ]


    # movable range for initial localization
    cov_threshold = 0.05
    move_range_x = (-0.00, 0.00)
    move_range_y = (-0.00, 0.00)

    use_zone_marker = True
    task_prompt = None
    ignore_threshold = True
    num_spin = 2
    TASK_INSTRUCT = "Switch the blue cube and red cube"
    # --------------------------------------- #

    navigator = Navigator(cov_threshold=cov_threshold, ignore_threshold=ignore_threshold,
                          num_spin=num_spin,
                          move_range_x=move_range_x, move_range_y=move_range_y)
    database = Database(left = landmark[0], middle=landmark[1], right=landmark[2])

    analyzer = ZoneCubeAnalyzer(
        image_path="/home/ubuntu/LLM_Planning/capture/capture.jpg",
        wait_before_capture=1.0,  # 필요하면 0.5나 0으로 조정 가능
    )

    grasping_node = GraspingNode("grasping")

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
            #head_up_q = [500, 700, 100, 250, 500]
            # higher
            head_up_q = [500, 800, 200, 250, 500]
            grasping_node.set_joint_positions_pulse_2(head_up_q, duration=2.0)
            time.sleep(5.0)

            # Database filling
            # Need to adjust camera pose if needed
            # zone_num, cube_color = analyzer()

            response = analyzer()
            if response in None:
                break
            else:
                zone_num = response[0]
                cube_color = response[1]

            print(f"[Gemini] direction={direction}, zone={zone_num}, cube={cube_color}")

            database.fill_zone_num(direction, zone_num)
            if cube_color is not None:
                database.fill_cube_info(cube_color, direction)

                # Move Arm to Marker Reading Position

                # Read Image - 이미지 어떻게 가져오지??

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
    environment_state = database.to_environment_state()
    print("Environment State:", environment_state)
    task_instruction = input(TASK_INSTRUCT)

    plan_text = create_robot_plan(environment_state, task_instruction)
    print("Gemini Response:\n", plan_text)
    
    task_seq = parse_final_plan(plan_text)
    print("Parsed Task Sequence:", task_seq)

    database.fill_task_seq(task_seq)

    print("\n------------------------------------\n")
    print("\t\tGemini Summary")
    print(database.zone_info)
    print(database.cube_info)
    print(database.dir_to_cube_info)
    print(database.task_seq)
    print("\n------------------------------------\n")

    cube_info_list = [("red", "left"),
                      ("green", "middle"),
                      ("blue", "right")]

    for info in cube_info_list:
        database.fill_cube_info(info[0], info[1])
        database.fill_dir_to_cube_info(info[1], info[0])

    print("\n------------------------------------\n")
    print("\t\tHand-added Info Summary")
    print(database.cube_info)
    print(database.dir_to_cube_info)
    print("\n------------------------------------\n")

# Just test workflow for grasping
    for direction in landmark_visit_order:
        navigation_is_done = navigator.set_goal(database.landmark[direction])
        if navigation_is_done:
            print(f"Arrived at Landmark: {direction}")
            cube_color = database.dir_to_cube_info[direction]
            print(f"Picking up: {cube_color}")
            success = grasping_node.grasp(cube_color)
            print("Pick and Place succeed")
        else:
            print(f"Failed to go to Landmark: {direction}")
            assert False



    # todo: database 형식 바꿔서 grasping node 에 먹여주기
    # todo: MPPI local minima escape

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
