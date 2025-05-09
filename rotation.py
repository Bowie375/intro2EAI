import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """
    Normalize the quaternion.

    Parameters
    ----------
    q: np.ndarray
        Unnormalized quaternion with shape (4,)

    Returns
    -------
    np.ndarray
        Normalized quaternion with shape (4,)
    """

    return q / np.linalg.norm(q)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """
    Return the conjugate of the quaternion.

    Parameters
    ----------
    q: np.ndarray
        Quaternion with shape (4,)

    Returns
    -------
    np.ndarray
        The conjugate of the quaternion with shape (4,)
    """

    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply the two quaternions.

    Parameters
    ----------
    q1, q2: np.ndarray
        Quaternions with shape (4,)

    Returns
    -------
    np.ndarray
        The multiplication result with shape (4,)
    """

    w1, v1 = q1[0], q1[1:]
    w2, v2 = q2[0], q2[1:]

    w = w1 * w2 - np.dot(v1, v2)
    v = w1 * v2 + w2 * v1 + np.cross(v1, v2)

    return np.concatenate(([w], v))

def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Use quaternion to rotate a 3D vector.

    Parameters
    ----------
    q: np.ndarray
        Quaternion with shape (4,)
    v: np.ndarray
        Vector with shape (3,)

    Returns
    -------
    np.ndarray
        The rotated vector with shape (3,)
    """

    q_conj = quat_conjugate(q)
    v_quat = np.concatenate(([0], v))
    q_rot = quat_multiply(quat_multiply(q, v_quat), q_conj)
    return q_rot[1:]

def quat_relative_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Compute the relative rotation angle between the two quaternions.

    Parameters
    ----------
    q1, q2: np.ndarray
        Quaternions with shape (4,)

    Returns
    -------
    float
        The relative rotation angle in radians, greater than or equal to 0.
    """
    
    q1_norm = quat_normalize(q1)
    q2_norm = quat_normalize(q2)

    d = np.dot(q1_norm, q2_norm)
    if d < 0:
        d = -d
        q2_norm = -q2_norm

    angle = 2 * np.arccos(d)
    return angle


def interpolate_quat(q1: np.ndarray, q2: np.ndarray, ratio: float) -> np.ndarray:
    """
    Interpolate between two quaternions with given ratio.

    When the ratio is 0, return q1; when the ratio is 1, return q2.

    The interpolation should be done in the shortest minor arc connecting the quaternions on the unit sphere.

    If there are multiple correct answers, you can output any of them.

    Parameters
    ----------
    q1, q2: np.ndarray
        Quaternions with shape (4,)
    ratio: float
        The ratio of interpolation, should be in [0, 1]

    Returns
    -------
    np.ndarray
        The interpolated quaternion with shape (4,)

    Note
    ----
    What should be done if the inner product of the quaternions is negative?
    """

    q1_norm = quat_normalize(q1)
    q2_norm = quat_normalize(q2)

    d = np.dot(q1_norm, q2_norm)
    if d < 0:
        d = -d
        q2_norm = -q2_norm

    angle = np.arccos(np.dot(q1_norm, q2_norm))    
    q_interp = (np.sin((1 - ratio) * angle) / np.sin(angle)) * q1_norm + (np.sin(ratio * angle) / np.sin(angle)) * q2_norm

    return q_interp


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """
    Convert the quaternion to rotation matrix.

    Parameters
    ----------
    q: np.ndarray
        Quaternion with shape (4,)

    Returns
    -------
    np.ndarray
        The rotation matrix with shape (3, 3)
    """

    theta = 2 * np.arccos(q[0])
    if theta < 1e-6:
        return np.eye(3)
    wx, wy, wz = q[1:] / np.sin(theta / 2)
    w_skew = np.array([
        [0, -wz, wy],
        [wz, 0, -wx],
        [-wy, wx, 0]
    ])
    return np.eye(3) + np.sin(theta) * w_skew + (1 - np.cos(theta)) * w_skew @ w_skew


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    """
    Convert the rotation matrix to quaternion.

    Parameters
    ----------
    mat: np.ndarray
        The rotation matrix with shape (3, 3)

    Returns
    -------
    np.ndarray
        The quaternion with shape (4,)
    """

    theta = np.arccos((np.trace(mat) - 1) / 2)
    if theta < 1e-6:
        return np.array([1, 0, 0, 0])
    w_skew = (1/(2*np.sin(theta)))*(mat - mat.T)
    w = np.array([w_skew[2, 1], w_skew[0, 2], w_skew[1, 0]])
    return np.concatenate(([np.cos(theta/2)], np.sin(theta/2) * w))


def quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """
    Convert the quaternion to axis-angle representation.

    The length of the axis-angle vector should be less or equal to pi.

    If there are multiple answers, you can output any.

    Parameters
    ----------
    q: np.ndarray
        The quaternion with shape (4,)

    Returns
    -------
    np.ndarray
        The axis-angle representation with shape (3,)
    """

    if q[0] < 0:
        q = -q

    theta = 2 * np.arccos(q[0])
    if theta < 1e-6:
        return np.zeros(3)
    axis = q[1:] / np.sin(theta / 2)
    return axis * theta


def axis_angle_to_quat(aa: np.ndarray) -> np.ndarray:
    """
    Convert the axis-angle representation to quaternion.

    The length of the axis-angle vector should be less or equal to pi

    Parameters
    ----------
    aa: np.ndarray
        The axis-angle representation with shape (3,)

    Returns
    -------
    np.ndarray
        The quaternion with shape (4,)
    """

    theta = np.linalg.norm(aa)
    if theta < 1e-6:
        return np.array([1, 0, 0, 0])
    w = aa / theta
    return np.concatenate(([np.cos(theta/2)], np.sin(theta/2) * w))

def axis_angle_to_mat(aa: np.ndarray) -> np.ndarray:
    """
    Convert the axis-angle representation to rotation matrix.

    The length of the axis-angle vector should be less or equal to pi

    Parameters
    ----------
    aa: np.ndarray
        The axis-angle representation with shape (3,)

    Returns
    -------
    np.ndarray
        The rotation matrix with shape (3, 3)
    """
    return quat_to_mat(axis_angle_to_quat(aa))


def mat_to_axis_angle(mat: np.ndarray) -> np.ndarray:
    """
    Convert the rotation matrix to axis-angle representation.

    The length of the axis-angle vector should be less or equal to pi

    Parameters
    ----------
    mat: np.ndarray
        The rotation matrix with shape (3, 3)

    Returns
    -------
    np.ndarray
        The axis-angle representation with shape (3,)
    """
    return quat_to_axis_angle(mat_to_quat(mat))


def uniform_random_quat() -> np.ndarray:
    """
    Generate a random quaternion with uniform distribution.

    Returns
    -------
    np.ndarray
        The random quaternion with shape (4,)
    """

    q = np.random.randn(4)
    q = q / np.linalg.norm(q)
    return q

def rpy_to_mat(rpy: np.ndarray) -> np.ndarray:
    """
    Convert roll-pitch-yaw euler angles into rotation matrix.

    This is required since URDF use this as rotation representation.

    Parameters
    ----------
    rpy: np.ndarray
        The euler angles with shape (3,)

    Returns
    -------
    np.ndarray
        The rotation matrix with shape (3, 3)
    """
    roll, pitch, yaw = rpy

    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    R = R_z @ R_y @ R_x  # Matrix multiplication in ZYX order
    return R
