import rclpy
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from std_srvs.srv import Empty
from tf2_ros import Buffer, TransformListener
from rclpy.duration import Duration
import math
import time


class Navigator():
    def __init__(self, cov_threshold, ignore_threshold, num_spin=1, move_range_x=(0.0, 0.0), move_range_y=(0.0, 0.0)):
        if not rclpy.ok():
            rclpy.init()

        self.navigator = BasicNavigator()
        self.start_point = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.navigator)

        # ADD !!
        # save starting point -> convert to map world frame coordinate
        # relocalizaiton..
        # if relocalization fail -> assert!
        # range of movement

        # initial covariance
        self.cov_threshold = cov_threshold
        self.ignore_threshold = ignore_threshold
        self.num_spin = num_spin
        self.move_range_x = move_range_x
        self.move_range_y = move_range_y

        self.cov_x = 999.0
        self.cov_y = 999.0
        self.cov_yaw = 999.0
        self.pose_sub = self.navigator.create_subscription(PoseWithCovarianceStamped,
                                                           '/amcl_pose',
                                                           self._amcl_callback,
                                                           10)

        # Initial Pose assumption
        initial_pose = PoseStamped()
        initial_pose.header.frame_id = 'map'
        initial_pose.header.stamp = self.navigator.get_clock().now().to_msg()
        initial_pose.pose.position.x = 0.0
        initial_pose.pose.position.y = 0.0
        initial_pose.pose.position.z = 0.0
        initial_pose.pose.orientation.w = 1.0
        initial_pose.pose.orientation.x = 0.0
        initial_pose.pose.orientation.y = 0.0
        initial_pose.pose.orientation.z = 0.0
        self.navigator.setInitialPose(initial_pose)

        print("Waiting Nav2...")
        self.navigator.waitUntilNav2Active()
        print("Done connecting Nav2. Ready to move.")

        if self.localize():
            print(">>> [Navigator] Initial Localization Completed Successfully.")
        else:
            print(">>> [Navigator] Warning: Localization Failed.")

        # update start point coordinate in map frame
        self._update_start_point()

    def _amcl_callback(self, msg):
        # called every data-subscription from amcl topic
        self.cov_x = msg.pose.covariance[0]
        self.cov_y = msg.pose.covariance[7]
        self.cov_yaw = msg.pose.covariance[35]
    def localize(self, max_retries=1):
        max_connect_retry = 30

        print("\n\t[Starting Localization...]")
        reinit = self.navigator.create_client(Empty, '/reinitialize_global_localization')

        while not reinit.wait_for_service(timeout_sec=2.0):
            print("Waiting for ACML node connection...")
        print("Connected to ACML node.")

        for i in range(max_connect_retry):
            req = Empty.Request()
            future = reinit.call_async(req)
            rclpy.spin_until_future_complete(self.navigator, future, timeout_sec=5.0)
            if future.done():
                # Done spreading particle
                print("Random Particles spread in map.")
                break
            else:
                # service no response
                print("Waiting for ACML particle-spreading service...")


        if self._check_converge():
            return True

        # move for update covariance
        for attempt in range(max_retries):
            print("Start Moving to localize")
            time.sleep(2.0)

            print(f"\n[Attempt {attempt+1}]")

            # Just Spin
            print(">>Spinning in current position...")
            print(f"Cov_x: {self.cov_x:.4f}"
                  f"\nCov_y: {self.cov_y:.4f}"
                  f"\nCov_yaw: {self.cov_yaw:.4f}")

            if self._spin_and_check():
                # self._update_start_point()
                self.move_to_start()
                return True

            # Move and Spin
            print(">>Moving in safe boundary...")
            print(f"Cov_x: {self.cov_x:.4f}"
                  f"\nCov_y: {self.cov_y:.4f}"
                  f"\nCov_yaw: {self.cov_yaw:.4f}")

            if self._move_and_check():
                # self._update_start_point()
                self.move_to_start()
                return True

        print("Failed to Localize.")
        return False

    def _check_converge(self, time_wait=0.1):
        rclpy.spin_once(self.navigator, timeout_sec=time_wait)
        done_update = (self.cov_x < self.cov_threshold) and \
                      (self.cov_y < self.cov_threshold) and \
                      (self.cov_yaw < self.cov_threshold)

        if self.ignore_threshold:
            print(f"Cov_x: {self.cov_x:.4f}\nCov_y: {self.cov_y:.4f}\nCov_yaw: {self.cov_yaw:.4f}")
            return False

        if done_update:
            print(f"Done Localization. Cov Threshold: {self.cov_threshold}")
            print(f"Cov_x: {self.cov_x:.4f}\nCov_y: {self.cov_y:.4f}\nCov_yaw: {self.cov_yaw:.4f}")
            return True
        else:
            return False

    def _spin_and_check(self):
        self.navigator.spin(spin_dist=3.14*2*self.num_spin, time_allowance=30)
        while not self.navigator.isTaskComplete():
            time.sleep(0.5)

        # check if spinning 360 is done well
        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            print("Done Spinning 360.")
        elif result == TaskResult.CANCELED:
            print("Spinning Cancelled.")
            assert False
        elif result == TaskResult.FAILED:
            print("Failed to Spin. Obstacle or Timelimit.")
            assert False

        if self._check_converge():
            print("Done Localization after spinning")
            return True
        else:
            return False

    def _move_and_check(self):
        # while moving in square shape,
        # check localization convergence
        # return to init pose
        goal_point_list = [(self.move_range_x[0], 0.0),   # min
                           (self.move_range_x[1], 0.0),   # max
                           (0.0, 0.0),
                           (0.0, self.move_range_y[0]),
                           (0.0, self.move_range_y[1])]

        goal = PoseStamped()
        goal.header.frame_id = 'odom'  # spawn frame
        goal.pose.position.z = 0.0
        goal.pose.orientation.w = 1.0
        goal.pose.orientation.x = 0.0
        goal.pose.orientation.y = 0.0
        goal.pose.orientation.z = 0.0

        for curr_point in goal_point_list:
            goal.header.stamp = self.navigator.get_clock().now().to_msg()
            goal.pose.position.x = curr_point[0]
            goal.pose.position.y = curr_point[1]

            self.navigator.goToPose(goal)
            while not self.navigator.isTaskComplete():
                if self._check_converge(time_wait=0.5):
                    print("Done Localization while moving.")
                    return True
            if self._check_converge():
                print("Done Localization while moving.")
                return True
        return False

    def _update_start_point(self):
        # START HERE
        try:
            # lookup_transform(target_frame, source_frame, time)
            t = self.tf_buffer.lookup_transform(
                'map',
                'odom',
                rclpy.time.Time(),
                timeout=Duration(seconds=1.0)
            )

            tx = t.transform.translation.x
            ty = t.transform.translation.y
            qx = t.transform.rotation.x
            qy = t.transform.rotation.y
            qz = t.transform.rotation.z
            qw = t.transform.rotation.w

            siny_cosp = 2 * (qw * qz + qx * qy)
            cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            yaw_deg = math.degrees(yaw)

            print(f"\n>>> Starting Point in Map frame:")
            print(f"X={tx:.3f}, Y={ty:.3f}, Yaw={yaw_deg:.1f}(deg)")
            self.start_point = (tx, ty, yaw_deg)

        except Exception as e:
            print(f"[TF Error] Failed to bring Transformation matrix: {e}")

    def move_to_start(self):
        # todo: [0,0,0] or self.start_point
        self.set_goal([0.0, 0.0, 0.0], mode='degrees', frame='odom')

    def set_goal(self, coordinate, mode='degrees', frame='map'):
        x = coordinate[0]
        y = coordinate[1]
        yaw = coordinate[2]

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = frame   # world frame coordinate
        goal_pose.header.stamp = self.navigator.get_clock().now().to_msg()

        # x, y in meter scale
        goal_pose.pose.position.x = float(x)
        goal_pose.pose.position.y = float(y)
        goal_pose.pose.position.z = 0.0

        if mode == 'degrees':
            yaw_rad = math.radians(yaw)
        elif mode == 'radians':
            yaw_rad = yaw
        else:
            assert False
        goal_pose.pose.orientation.x = 0.0
        goal_pose.pose.orientation.y = 0.0
        goal_pose.pose.orientation.z = math.sin(yaw_rad / 2.0)
        goal_pose.pose.orientation.w = math.cos(yaw_rad / 2.0)

        print(f"Start Navigation: x={x}, y={y}, yaw={yaw}")
        self.navigator.goToPose(goal_pose)

        while not self.navigator.isTaskComplete():
            feedback = self.navigator.getFeedback()

            # feedback이 None일 수도 있으므로 체크 필요
            if feedback:
                remain_dist = feedback.distance_remaining
                recoveries = feedback.number_of_recoveries
                elapsed_time = feedback.navigation_time.sec

                print(f'[이동중] 남은 거리: {remain_dist:.2f}m | '
                      f'경과 시간: {elapsed_time}초 | '
                      f'회복 시도: {recoveries}회')

            time.sleep(0.5) # 0.5초마다 로그 출력

        result = self.navigator.getResult()
        if result == TaskResult.SUCCEEDED:
            print("Done Navigation. Returning to Core Process...")
            return True
        else:
            print("Failed to Navigate. Returning to Core Process...")
            return False

    def stop(self):
        self.navigator.cancelTask()