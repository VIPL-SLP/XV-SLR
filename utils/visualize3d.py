import torch
import numpy as np

class CocoViewer(object):
    def __init__(self) -> None:
        self.connection = {}
        self.connection['pose'] = ([11, 12],
            [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
            [8, 10], [1, 2], [0, 1], [0, 2],
            [1, 3], [2, 4], [3, 5], [4, 6], [15, 17])
        self.connection['left_hand'] = ([91, 92],
            [92, 93], [93, 94], [94, 95], [91, 96], [96, 97],
            [97, 98], [98, 99], [91, 100], [100, 101], [101, 102],
            [102, 103], [91, 104], [104, 105], [105, 106],
            [106, 107], [91, 108], [108, 109], [109, 110],
            [110, 111])
        self.connection['right_hand'] = ([112, 113], [113, 114], [114, 115],
            [115, 116], [112, 117], [117, 118], [118, 119],
            [119, 120], [112, 121], [121, 122], [122, 123],
            [123, 124], [112, 125], [125, 126], [126, 127],
            [127, 128], [112, 129], [129, 130], [130, 131],
            [131, 132])
        self.skip_points = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22] + list(range(41, 92))

        self.color_mapper = {
            'pose': (255, 0, 0),
            'left_hand': (0, 255, 0),
            'right_hand': (0, 0, 255)
        }
    
    def draw_single_image(self, kps, img):
        kps[:,0] -= 52
        kps[:,:2] /= (408/256)
        for index in range(len(kps)):
            if index in self.skip_points:
                continue
            if index < 23:
                color = (255, 0, 0)
            elif index < 91:
                color = (255, 255, 255)
            elif index < 112:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.circle(img, (int(kps[index][0]), int(kps[index][1])), 1, color, 1)
        for k, v in self.connection.items():
            for connection in v:
                start_idx, end_idx = connection
                if start_idx in self.skip_points or end_idx in self.skip_points:
                    continue
                point1 = (int(kps[start_idx][0]), int(kps[start_idx][1]))
                point2 = (int(kps[end_idx][0]), int(kps[end_idx][1]))
                cv2.line(img, point1, point2, self.color_mapper[k], 1)
        return img

    def draw(self, skeleton, image_list):
        if isinstance(skeleton, str):
            skeleton = np.load(skeleton)
        skeleton[:,:,0] -= 52
        skeleton[:,:,:2] /= (408/256)
        assert len(skeleton) == len(image_list)
        for i in range(len(image_list)):
            image_list[i] = self.draw_single_image(skeleton[i], image_list[i])
        return image_list

def perspective_projection(points, rotation, translation, focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs, 2) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:, 0, 0] = focal_length[:, 0]
    K[:, 1, 1] = focal_length[:, 1]
    K[:, 2, 2] = 1.
    K[:, :-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]

def get_rotation(theta_x=0, theta_y=0, theta_z=0):
    """
    Applies an additional rotation around the x-axis to a rotation matrix.
    
    Parameters:
        R_original: 3x3 numpy array, the original rotation matrix.
        theta: float, the rotation angle in radians along the x-axis.
    
    Returns:
        R_new: 3x3 numpy array, the updated rotation matrix.
    """
    R_axis_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])
    R_axis_y = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])
    R_axis_z = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])
    
    return R_axis_z @ R_axis_y @ R_axis_x