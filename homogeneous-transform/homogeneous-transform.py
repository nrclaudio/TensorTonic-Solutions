import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    T = np.asarray(T)
    points = np.asarray(points)
    if len(points.shape) < 2:
        points_homogeneous = np.append(points, np.array([1]))
        transformed_points = T @ points_homogeneous
        spatial = transformed_points[:3].copy()
    else:
        ones = np.ones((points.shape[0], 1))
        points_homogeneous = np.append(points,ones, axis=1)
        transformed_points = points_homogeneous @ T.T
        spatial = transformed_points[:, :3].copy()
    return spatial
    
    
