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

    # ------ map_nav_test --------- #
    w1 = (1.069, 0.470, 81.401)
    p1_right = (1.130, 0.911, 81.401)
    p1 = (1.130, 0.911, 178.548)
    p1_back = (1.130, 0.911, 0.0)
    w2_left = (1.672, 0.919, 1.845)
    w2 = (1.672, 0.919, 1.845)
    p2 = (1.762, 0.420, -86.034)
    p2_back = (1.762, 0.420, 90.0)
    w2_back = (1.672, 0.919, 90.0)
    w2_right = (1.672, 0.919, 90.0)
    w1_back = (1.069, 0.470, -90.0)


    landmark = [w1, p1_right, p1, p1_back, w2_left, w2, p2, p2_back, w2_back, w2_right, w1_back ]

    p1_path = []
    # ------ map nav_test --------- #

    num_spin = 2
    use_zone_marker = True
    task_prompt = None
    ignore_threshold = True
    TASK_INSTRUCT = "Switch the blue cube and red cube"
    # --------------------------------------- #

    navigator = Navigator(num_spin=num_spin)
    database = Database(left = landmark[0], middle=landmark[1], right=landmark[2])
    analyzer = ZoneCubeAnalyzer(
        image_path="/home/ubuntu/LLM_Planning/capture/capture.jpg",
        wait_before_capture=1.0,  # 필요하면 0.5나 0으로 조정 가능
    )
    grasping_node = GraspingNode("grasping")


    start_point_curr = navigator.start_point
    assert  start_point_curr is not None


    for idx, way_point in enumerate(landmark):
        navigation_is_done = navigator.set_goal(way_point)
        if navigation_is_done:
            print(f"Arrived at Way Point {idx+1}")
        else:
            print(f"Failed to go to Way Point {idx+1}")

    navigator.move_to_start()

    time.sleep(300.0)
    assert False






    # Visiting Order
    landmark_dist = {'left': -1.0, 'middle': -1.0, 'right':-1.0}

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
