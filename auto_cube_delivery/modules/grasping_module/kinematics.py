import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from scipy.optimize import least_squares
from .utils.kinematics_utils import get_urdf
from jaxlie import SO3

#####

def forward_kinematics(
    q: np.ndarray, 
    target: str,
) -> np.ndarray:
    """
    Compute the forward kinematics for the robotic arm.
    Args:
        q (np.ndarray): Joint angles (1D array of size 5) in radian.
        target (str): Target frame to compute the pose for. Options are:
                      'j1', 'j2', 'j3', 'j4', 'j5', 'cam', 'tcp'.
                      Default is 'j1'.
    Returns:
        jnp.ndarray: 4x4 transformation matrix of the target frame.
    """

    ######
    ## TODO : Implement Forward Kinematics using the robot's URDF parameters.
    ## You can refer to the URDF parameters defined in get_urdf() function in utils/kinematics_utils.py.
    urdf = get_urdf()
    q = jnp.array(q)
    
    def axis_angle_to_matrix(axis, angle):
        axis = jnp.array(axis)
        axis = axis / jnp.linalg.norm(axis)

        # using skew-symmetric matrix
        K = jnp.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]], 
            [-axis[1], axis[0], 0]
        ])

        R = jnp.eye(3) + jnp.sin(angle) * K + (1 - jnp.cos(angle)) * K @ K

        # using quaternion representation
        # so3_rot = SO3.from_quaternion_xyzw(
        #     jnp.array([
        #         axis[0] * jnp.sin(angle/2),
        #         axis[1] * jnp.sin(angle/2), 
        #         axis[2] * jnp.sin(angle/2),
        #         jnp.cos(angle/2)
        #     ])
        # )
        # R = so3_rot.as_matrix()

        T = jnp.eye(4)
        T = T.at[:3, :3].set(R)
        return T
    
    T_base = jnp.array(urdf["T_base"])
    T_joint1 = jnp.array(urdf["T_base_joint_1"])
    T_joint2 = jnp.array(urdf["T_joint1_joint_2"])
    T_joint3 = jnp.array(urdf["T_joint2_joint_3"])
    T_joint4 = jnp.array(urdf["T_joint3_joint_4"])
    T_joint5 = jnp.array(urdf["T_joint4_joint_5"])
    T_TCP = jnp.array(urdf["T_joint5_TCP_offset"])
    T_cam = jnp.array(urdf["T_joint4_cam_offset"])
    
    R1 = axis_angle_to_matrix(urdf["Axis_joint_1"], q[0])
    R2 = axis_angle_to_matrix(urdf["Axis_joint_2"], q[1])
    R3 = axis_angle_to_matrix(urdf["Axis_joint_3"], q[2])
    R4 = axis_angle_to_matrix(urdf["Axis_joint_4"], q[3])
    R5 = axis_angle_to_matrix(urdf["Axis_joint_5"], q[4])

    T01 = T_base @ T_joint1 @ R1
    T02 = T01 @ T_joint2 @ R2
    T03 = T02 @ T_joint3 @ R3
    T04 = T03 @ T_joint4 @ R4
    T05 = T04 @ T_joint5 @ R5
    
    if target == 'j1':
        pose = T01
    elif target == 'j2':
        pose = T02
    elif target == 'j3':
        pose = T03
    elif target == 'j4':
        pose = T04
    elif target == 'j5':
        pose = T05
    elif target == 'tcp':
        pose = T05 @ T_TCP
    elif target == 'cam':
        pose = T04 @ T_cam
    else:
        pose = None
        raise ValueError(f"Invalid target: {target}.")

    #####

    return pose

def inverse_kinematics(
    q: np.ndarray, 
    T_target: np.ndarray
) -> dict | None:
    """
    Compute the inverse kinematics for the robotic arm.
    Args:
        q (np.ndarray): Initial joint angles (1D array of size 5) in radian.
        T_target (np.ndarray): 4x4 transformation matrix of the target end-effector pose (tcp).
    
        Returns:
            dict | None: A dictionary containing the solution joint angles and position error, or None if failed.
                {
                    "sol": np.ndarray,  # Solution joint angles (1D array of size 5) in radian or None if failed
                    "pos_error": float  # Position error in meters or None if failed
                }
    """

    #####
    ## TODO : Implement Inverse Kinematics using numerical optimization.
    ## You may use jax.jacfwd, scipy.optimize.least_squares and forward_kinematics function for this purpose.
    ## When the optimization fails, return None.
    ## (Optional) Refer to jax.jit function to speed up the repeated computation.

    pos_error : float = None
    ik_solution : np.ndarray = None

    q = jnp.array(q)
    T_target = jnp.array(T_target)

    eps = np.random.uniform(low=-1e-6, high=1e-6, size=q.shape)

    def residual(q, T_target):
        # calc current pose
        T_current = forward_kinematics(q, target='tcp')
        
        pos_error = T_current[:3, 3] - T_target[:3, 3]
        
        R_current = T_current[:3, :3]
        R_target = T_target[:3, :3]

        R_current_so3 = SO3.from_matrix(R_current)
        R_target_so3 = SO3.from_matrix(R_target)
        R_error_so3 = R_current_so3.inverse() @ R_target_so3
        rot_error_3d = R_error_so3.log()   

        # Return 6D residual: [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z]
        return jnp.concatenate([pos_error, rot_error_3d])
    
    # Jacobian function using JAX
    residual_jac = jax.jacfwd(residual, argnums=0)

    result = least_squares(
        fun=lambda q_opt: np.array(residual(jnp.array(q_opt), T_target)),
        x0=np.array(q),
        jac=lambda q_opt: np.array(residual_jac(jnp.array(q_opt), T_target)),
        xtol=2.23e-16
    )
    
    if result.success:
        ik_solution = result.x
        T_final = forward_kinematics(ik_solution, 'tcp')
        # print("T_final")
        # print(T_final)

        pos_error = float(jnp.linalg.norm(T_final[:3, 3] - T_target[:3, 3]))

    else:
        ik_solution = None
        pos_error = None
    #####

    result = {}
    result["sol"] = ik_solution
    result["pos_error"] = pos_error

    return result
