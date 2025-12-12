import numpy as np
import time
import cv2
import rclpy

from .kinematics import *
from .utils.kinematics_utils import *
from .utils.grasping_base import GraspingNodeBase
from .run_pose_estimation import CubePoseEstimator
import message_filters
from sensor_msgs.msg import Image
from collections import deque
from cv_bridge import CvBridge

class GraspingNode(GraspingNodeBase):
    def __init__(self, name):
        super().__init__(name)

        self.running = True
        
        self.last_q = None
        self.grasping = False
        self.grasp_targets = []
        self.interpolation_error = 0
        self.T_base_marker = None
        self.bridge = CvBridge()
        self.num_collected = 0
        
        # --- State Variables ---
        self.is_collecting = False  # True일 때만 Vision 데이터 수집
        self.pose_history = deque(maxlen=50) # 50개 샘플 저장소
        self.last_detected_pose = None # 최종 계산된 평균 Pose (4x4)

        # --- Vision Settings ---
        self.target_color = None
        self.cube_size = 0.03
        self.calib_path = 'resources/camera_calibration.npz'
        
        self.color_ranges = {
            'red':   [(np.array([0, 100, 50]), np.array([10, 255, 255])),
                      (np.array([170, 100, 50]), np.array([180, 255, 255]))],
            'green': [(np.array([40, 80, 50]), np.array([90, 255, 255]))],
            'blue':  [(np.array([100, 100, 50]), np.array([140, 255, 255]))]
        }

        # --- Estimator Init ---
        try:
            self.estimator = CubePoseEstimator(self.cube_size, self.calib_path)
            self.get_logger().info("PoseEstimator Loaded.")
        except Exception as e:
            self.get_logger().error(f"Estimator Load Failed: {e}")
            exit(-1)

        # Vision Sync (RGB + Depth)
        rgb_sub = message_filters.Subscriber(self, Image, '/depth_cam/rgb/image_raw')
        depth_sub = message_filters.Subscriber(self, Image, '/depth_cam/depth/image_raw')
        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.vision_callback)
        #####
        ## TODO : Define custom class variables, if needed.
        #####

    ############################################
    # Rotation / Pose interpolation helpers
    ############################################

    @staticmethod
    def rot_x(angle: float) -> np.ndarray:
        """
        로컬 x축 기준 회전 행렬 (3x3)
        """
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c]
        ], dtype=float)

    @staticmethod
    def rot_y(angle: float) -> np.ndarray:
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([
            [ c, 0.0,  s],
            [0.0, 1.0, 0.0],
            [-s, 0.0,  c],
        ], dtype=float)
    
    @staticmethod
    def rot_z(angle: float) -> np.ndarray:
        """
        로컬 z축 기준 회전 행렬 (3x3)
        """
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=float)

    def detect_color_bbox(self, cv_image):
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
            ranges = self.color_ranges.get(self.target_color)
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for (lower, upper) in ranges: mask += cv2.inRange(hsv, lower, upper)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: return None
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) < 300: return None 
            x, y, w, h = cv2.boundingRect(c)
            pad = 5
            h_img, w_img = cv_image.shape[:2]
            return [max(0, x-pad), max(0, y-pad), min(w_img, x+w+pad), min(h_img, y+h+pad)]

    def find_best_ik_around_tcp(
        self,
        q_seed: np.ndarray,
        p_target: np.ndarray,
        num_samples: int = 12,
        y_direction: float = 0.0
    ):
        """
        - q_seed: 시작 조인트 (보통 self.last_q)
        - p_target: base frame에서의 목표 위치 (마커 위치)
        - num_pitch_samples: pitch를 몇 개로 샘플링할지 (12면 30도 간격)

        반환:
            best_ik_sol (dict) 혹은 None
        """
        # 현재 tcp pose
        T_curr = forward_kinematics(q_seed, target='tcp')
        R_curr = T_curr[:3, :3]

        best = None
        best_err = 1e9
        
        best_roll = 0
        best_pitch = 0
        best_yaw = 0
        
        pitch_updated = False
        yaw_updated = False
        
        step = 0

        print("----------Roll Sampling----------")
        for roll in np.linspace(-np.pi/6, np.pi/6, num_samples, endpoint=False):
            if y_direction <= 0:
                roll = -roll
            roll_updated = False
            step += 1
            R_candidate = R_curr @ self.rot_x(roll)   # tcp z축 기준 pitch 회전

            T_candidate = np.eye(4)
            T_candidate[:3, :3] = R_candidate
            T_candidate[:3, 3] = p_target
            
            print(f"STEP in Roll: {step}")

            ik = inverse_kinematics(q_seed, T_candidate)
            if not self.is_ok(ik):
                continue

            pos_err = ik["pos_error"] * 100
            print(f"POS ERROR in steps: {pos_err:.3f} cm")
            # 필요하면 fk 기준으로 한 번 더 검증해도 됨
            # T_fk = forward_kinematics(ik["sol"], target='tcp')
            # pos_err = np.linalg.norm(T_fk[:3,3] - p_target)

            if pos_err < best_err:
                best_err = pos_err
                best = ik
                best_roll = roll * 180 / np.pi
                roll_updated = True
            
            if not roll_updated:
                print(f"Find the best Roll: {best_roll:.1f} degree")
                break
        
        T_best = forward_kinematics(best["sol"], target='tcp')
        R_best = T_best[:3, :3]
        
        best = None
        best_err = 1e9
        step = 0

        print("----------Pitch Sampling----------")
        # 5 degree ~ 60 degree 범위에서 pitch 샘플링
        for pitch in np.linspace(np.pi/72, np.pi/3, num_samples, endpoint=False):
            pitch_updated = False
            step += 1
            R_candidate = R_best @ self.rot_y(pitch)   # tcp z축 기준 pitch 회전

            T_candidate = np.eye(4)
            T_candidate[:3, :3] = R_candidate
            T_candidate[:3, 3] = p_target
            
            print(f"STEP in Pitch: {step}")

            ik = inverse_kinematics(q_seed, T_candidate)
            if not self.is_ok(ik):
                continue

            pos_err = ik["pos_error"] * 100
            print(f"POS ERROR in steps: {pos_err:.3f} cm")

            if pos_err < best_err:
                best_err = pos_err
                best = ik
                best_pitch = pitch * 180 / np.pi
                pitch_updated = True
            
            if not pitch_updated:
                print(f"Find the best Pitch: {best_pitch:.1f} degree")
                break

        return best, best_err, best_roll, best_pitch

    # ---------------------------------------------------------
    # Vision Callback (Data Collection)
    # ---------------------------------------------------------
    def vision_callback(self, rgb_msg, depth_msg):
        # [중요] 수집 모드가 아니면 연산하지 않고 리턴 (CPU 절약)
        if not self.is_collecting:
            return

        try:
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        except: return

        bbox = self.detect_color_bbox(cv_rgb)
        if bbox:
            if cv_depth.dtype == np.uint16:
                depth_meter = cv_depth.astype(np.float32) * 0.001
            else:
                depth_meter = cv_depth
            
            # 1. T_cam_marker 계산
            T_cam_marker = self.estimator.estimate_pose(depth_meter, bbox)

            if T_cam_marker is not None and self.last_q is not None:
                try:
                    self.num_collected += 1
                    # 2. FK 계산 (Base 기준)
                    print(f"Collecting {self.num_collected} Samples")
                    T_base_cam = forward_kinematics(self.last_q, target='cam')
                    self.T_base_marker = T_base_cam @ T_cam_marker
                    
                    # 3. Translation 저장
                    self.pose_history.append(self.T_base_marker[:3, 3])
                    avg_pos = np.mean(np.array(self.pose_history), axis=0)
                    print(f"Target Base Position ({self.num_collected} sample avg): {avg_pos}")
                    # (Optional) 화면에 진행상황 표시
                    # cv2.imshow("Vision", cv_rgb)
                    # cv2.waitKey(1)
                except:
                    pass

    ############################################
    # 기존 utility / 제공된 메서드들
    ############################################
    def is_ok(self, ik_sol):
        return (ik_sol is not None and ik_sol["sol"] is not None and ik_sol["pos_error"] is not None)

    def mapping_marker_id(self, target_marker_id):
        marker_id_map = {
            "R": 1,
            "G": 7,
            "B": 6
        }
        if isinstance(target_marker_id, str):
            return marker_id_map.get(target_marker_id.upper(), -1)
        return target_marker_id

    ### Utility functions for controlling the robot arm and gripper
    ### DO NOT MODIFY THESE FUNCTIONS

    def gripper_close(self, duration=1.5):
        '''
        Close the gripper.
        '''
        self._set_position_pulse([(10, 550)], duration)

    def gripper_open(self, duration=1.5):
        '''
        Open the gripper.
        '''
        self._set_position_pulse([(10, 100)], duration)

    def get_joint_positions(self):
        '''
        Returns: current joint positions in "radians" as a numpy array
        '''
        q = self.get_joint_positions_pulse() # Base function method
        return pulse2angle(q)
    
    def set_joint_positions(self, q, duration):
        '''
        q: "radians", list or numpy array of joint angles
        duration: time to move in seconds
        '''
        pulse = angle2pulse(q)
        self.set_joint_positions_pulse(pulse, duration) # Base function method

    def set_joint_positions_pulse(self, pulse, duration):
        '''
        pulse: list or numpy array of joint pulses
        duration: time to move in seconds
        '''
        self.set_joint_positions_pulse(pulse, duration) # Base function method
    
    #########################################################################################################################
    ## TODO : Implement the following functions to complete the grasping functionality.

    def grasp(self, cube_color: int | str) -> bool:
        is_success = False
        self.target_color = cube_color
        try:
            # 현재 조인트 상태 저장
            default_q = [500, 700, 100, 250, 500]
            
            self.last_q = pulse2angle(default_q)
            self.set_joint_positions(self.last_q, duration=2.0)
            time.sleep(5.0)

            print("=== Step : Collecting 3 Pose Samples ===")
            self.pose_history.clear()
            self.is_collecting = True # Vision Callback 활성화
            
            # [핵심] 50개 모일 때까지 rclpy.spin_once로 콜백 강제 실행
            while len(self.pose_history) < 3:
                rclpy.spin_once(self, timeout_sec=0.01)
                # time.sleep(0.01) # spin_once가 blocking이 아니면 필요할 수 있음

            self.is_collecting = False 
            self.num_collected = 0
            # 수집 종료
            print("=== Data Collection Complete ===")

            # 평균 계산
            avg_pos = np.mean(np.array(self.pose_history), axis=0)
            print(f"Target Base Position (Avg): {avg_pos}")
            
            print("=== Step : Finding IK & Grasping ===")
            # 목표 위치 보정 (Z축 약간 위로)
            p_target = avg_pos.copy()

            # 그리퍼 오픈
            self.gripper_open(duration=1.0)
            time.sleep(1.0)

            # Tuning Start
            if p_target[0] > 0.42:
                p_target[0] += 0.015
            elif (p_target[0] < 0.42 and p_target[0] > 0.35):
                p_target[0] += 0.0075
            else:
                p_target[0] += 0.0025
            
            if p_target[1] < 0 : 
                p_target[1] -= 0.005
                if p_target[1] < -0.1 : 
                    p_target[1] -= 0.005

            if p_target[1] > 0 : 
                p_target[1] += 0.005
                if p_target[1] > 0.1 : 
                    p_target[1] += 0.005

            p_target[2] -= 0.045
            # Tuning End

            # 여기서 config 문제를 피하기 위해 pitch 여러 개를 찍어봄
            best_ik, best_err, best_roll, best_pitch = self.find_best_ik_around_tcp(
                q_seed=self.last_q,
                p_target=p_target,
                num_samples=20,  # 22.5도 간격 등으로 늘려도 됨
                y_direction = p_target[1]
            )

            if best_ik is None:
                raise RuntimeError("No feasible IK found around tcp pitch")
                
            q_target = best_ik["sol"]

            print("----------Best Results----------")                 
            print(f"[grasp] best pos_error: {best_err:.3f} cm")
            print(f"[grasp] best roll: {best_roll:.1f} degree")
            print(f"[grasp] best pitch: {best_pitch:.1f} degree")            
            
            T_target = forward_kinematics(q_target, target='tcp')
            
            print("----------T_base_marker----------")            
            print(self.T_base_marker)

            print("----------T_target----------")            
            print("T_target")
            print(T_target)
            
            self.set_joint_positions(q_target, duration=3.0)
            time.sleep(3.0)
            
            self.gripper_close(duration=1.0)
            time.sleep(1.5)

            # Back to Initial Position
            default_q = [500, 700, 100, 250, 500]
            
            self.last_q = pulse2angle(default_q)
            self.set_joint_positions(self.last_q, duration=2.0)
            time.sleep(5.0)

            # Check cube existance
            print("Checking Cube Existance...")
            self.pose_history.clear()
            self.is_collecting = True # Vision Callback 활성화
            
            start = time.time()
            while len(self.pose_history) < 1:
                rclpy.spin_once(self, timeout_sec=0.01)
                end = time.time()
                if end-start > 5.0:
                    print("=== Robot Grasps the Cube! ===")
                    self.is_collecting = False
                    break
            if (len(self.pose_history) > 0):
                print("=== Robot Misses the Cube! ===")
                self.num_collected = 0
                self.grasp(self.target_color)
            
            self.gripper_open(duration=1.0)
            time.sleep(1.5)
            
            is_success = True
            return is_success

        except Exception as e:
            print("Grasp action failed:", e)

        finally:
            # 성공/실패와 관계없이 원래 자세로 복귀 (원하면 주석 처리 가능)
            if self.last_q is not None:
                self.set_joint_positions(self.last_q, duration=3.0)

        return is_success

    def place(self, action_name) -> bool:
        """
            Execute placing action given an action name.
            You may utilize predefined action groups if available.
        """
        is_success = False
        try:
            self.controller.run_action(action_name)
            self.gripper_open(duration=1.0)
            # 다시 원래 자세로 복귀
            if self.last_q is not None:
                self.set_joint_positions(self.last_q, duration=1.0)
            is_success = True
        except Exception as e:
            print("Place action failed.", e)
            pass

        return is_success
