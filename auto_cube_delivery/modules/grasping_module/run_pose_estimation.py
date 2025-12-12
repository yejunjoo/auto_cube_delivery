import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import warnings
warnings.filterwarnings("ignore")

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import message_filters
import copy
import time

import cv2
import numpy as np
import open3d as o3d
from .kinematics import *
from .utils.kinematics_utils import *
from collections import deque

# =====================================================================
# 1. Pose Estimator Class (변경 없음)
# =====================================================================
class CubePoseEstimator:
    def __init__(self, cube_size, calib_file_path='resources/camera_calibration.npz'):
        self.cube_size = cube_size
        if not os.path.exists(calib_file_path):
            raise FileNotFoundError(f"Calibration file '{calib_file_path}' not found!")
        with np.load(calib_file_path) as X:
            self.K, self.dist_coeffs = [X[i] for i in ('mtx','dist')]
        
        self.target_model = self._create_ideal_cube_pcd(cube_size)

    def _create_ideal_cube_pcd(self, size, num_points=2000):
        mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
        mesh.translate((-size/2, -size/2, -size/2)) 
        pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=size*0.4, max_nn=40))
        return pcd

    def estimate_pose(self, depth_image_meter, bbox):
        xmin, ymin, xmax, ymax = bbox
        h, w = depth_image_meter.shape
        xmin = max(0, xmin); ymin = max(0, ymin)
        xmax = min(w, xmax); ymax = min(h, ymax)
        
        depth_crop = depth_image_meter[ymin:ymax, xmin:xmax].copy()
        mask = (depth_crop > 0.05) & (depth_crop < 1.5)
        if np.count_nonzero(mask) < 50: return None

        iy, ix = np.indices(depth_crop.shape)
        ix += xmin; iy += ymin
        z = depth_crop
        x = (ix - self.K[0, 2]) * z / self.K[0, 0]
        y = (iy - self.K[1, 2]) * z / self.K[1, 1]
        
        points = np.stack((x[mask], y[mask], z[mask]), axis=-1)
        if points.shape[0] < 50: return None

        scene_pcd = o3d.geometry.PointCloud()
        scene_pcd.points = o3d.utility.Vector3dVector(points)
        scene_pcd, _ = scene_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        if len(scene_pcd.points) < 30: return None
        scene_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.cube_size*0.4, max_nn=40))

        try:
            obb = scene_pcd.get_oriented_bounding_box()
            init_trans = np.eye(4)
            init_trans[:3, :3] = obb.R
            init_trans[:3, 3] = obb.center
        except: return None

        reg_p2p = o3d.pipelines.registration.registration_icp(
            self.target_model, scene_pcd, max_correspondence_distance=0.005,
            init=init_trans,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        if reg_p2p.fitness < 0.3: return None
        return reg_p2p.transformation

def draw_3d_axis(img, K, dist, pose, axis_len=0.05):
    if pose is None: return img
    points_3d = np.float32([[0,0,0], [axis_len,0,0], [0,axis_len,0], [0,0,axis_len]])
    r_vec, _ = cv2.Rodrigues(pose[:3, :3])
    t_vec = pose[:3, 3]
    points_2d, _ = cv2.projectPoints(points_3d, r_vec, t_vec, K, dist)
    points_2d = np.int32(points_2d).reshape(-1, 2)
    origin = tuple(points_2d[0])
    img = cv2.line(img, origin, tuple(points_2d[1]), (0,0,255), 3)
    img = cv2.line(img, origin, tuple(points_2d[2]), (0,255,0), 3)
    img = cv2.line(img, origin, tuple(points_2d[3]), (255,0,0), 3)
    return img

# =====================================================================
# 2. Main Node Class (JointState 추가 및 FK 계산)
# =====================================================================
class CubePoseNode(Node):
    def __init__(self):
        super().__init__('cube_pose_base_node', allow_undeclared_parameters=True, automatically_declare_parameters_from_overrides=True)
        self.get_logger().info('Starting Base-Frame Cube Pose Estimator...')
        self.bridge = CvBridge()
        self.estimator = None
        self.current_joints = None # 현재 관절 각도 저장용
        self.pose_history = deque(maxlen=50)

        # 설정
        self.target_color = 'red'  
        self.cube_size = 0.03
        self.calib_path = 'resources/camera_calibration.npz'

        self.color_ranges = {
            'red':   [(np.array([0, 100, 50]), np.array([10, 255, 255])),
                      (np.array([170, 100, 50]), np.array([180, 255, 255]))],
            'green': [(np.array([40, 80, 50]), np.array([90, 255, 255]))],
            'blue':  [(np.array([100, 100, 50]), np.array([140, 255, 255]))]
        }

        # 1. Estimator 초기화
        try:
            self.estimator = CubePoseEstimator(self.cube_size, self.calib_path)
            self.get_logger().info("Estimator Initialized.")
        except Exception as e:
            self.get_logger().error(f"Estimator Error: {e}")
            exit(-1)

        # 3. Image Sync Subscriber
        rgb_sub = message_filters.Subscriber(self, Image, '/depth_cam/rgb/image_raw')
        depth_sub = message_filters.Subscriber(self, Image, '/depth_cam/depth/image_raw')
        self.ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.sync_callback)

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

    def sync_callback(self, rgb_msg, depth_msg):
        try:
            cv_rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        except Exception as e: return

        output_image = cv_rgb.copy()
        bbox = self.detect_color_bbox(cv_rgb)
        
        if bbox:
            cv2.rectangle(output_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # --- T_cam_marker 구하기 ---
            if cv_depth.dtype == np.uint16:
                depth_meter = cv_depth.astype(np.float32) * 0.001
            else:
                depth_meter = cv_depth
            T_cam_marker = self.estimator.estimate_pose(depth_meter, bbox)
            
            if T_cam_marker is not None:
                print("Pose Estimation Done.")
                # 1. 화면에 카메라 기준 축 그리기
                # output_image = draw_3d_axis(output_image, self.estimator.K, self.estimator.dist_coeffs, T_cam_marker, axis_len=self.cube_size * 2)
                try:
                    last_q = [0.00418879, -1.10165182, 2.0525072, 1.17286126, 0.0]
                    # Get the current cam pose in base frame
                    T_base_cam = forward_kinematics(last_q, target='cam')
                    # Get the marker pose in base frame
                    T_base_marker = T_base_cam @ T_cam_marker
                    # Extract Translation (x, y, z)
                    bx, by, bz = T_base_marker[:3, 3]
                    self.pose_history.append([bx, by, bz])
                    
                    avg_x, avg_y, avg_z = np.mean(self.pose_history, axis=0)
                    
                    count = len(self.pose_history)
                    
                    print(f"Avg {count}: {avg_x:.3f}, {avg_y:.3f}, {avg_z:.3f}")
                    
                    # 화면에 출력 (Base 좌표)
                    text_base = f"Base: ({bx:.3f}, {by:.3f}, {bz:.3f})"
                    cv2.putText(output_image, text_base, (bbox[0], bbox[1]-25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2) # 빨간색 글씨
                                
                except Exception as e:
                    # FK 계산 에러 시 (import 안됐거나 등등)
                    cv2.putText(output_image, "FK Error", (bbox[0], bbox[1]-25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # 참고용: Camera 기준 좌표도 작게 표시
                cx, cy, cz = T_cam_marker[:3, 3]
                text_cam = f"Cam : ({cx:.2f}, {cy:.2f}, {cz:.2f})"
                cv2.putText(output_image, text_cam, (bbox[0], bbox[1]-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow("Base Frame Cube Pose", output_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()
            cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = CubePoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
