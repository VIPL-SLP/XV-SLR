import os
import pdb

import glob
import json
import GPUtil
import pickle
import matplotlib
matplotlib.use("Agg")
import numpy as np
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt
# from mediapipe_layer import MediaPipe
from scipy.optimize import linear_sum_assignment
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
from mmpose.utils import adapt_mmdet_pipeline
from mmpose.evaluation.functional import nms
from multiprocessing import Pool
import torch
import time
import torch.multiprocessing as mp
from mmdet.apis import inference_detector, init_detector


def mediapipe2coco(mp, img):
    body_idx = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 27, 27, 28]
    feet_idx = [31, 29, 27, 32, 30, 28]
    face_idx = [34, 227, 93, 213, 138, 136, 149, 148, 377, 400, 379, 365, 367, 435, 366, 447, 264, 70, 52, 66, 107, 55, 336, 295, 282, 283, 276, 168, 351, 248, 363, 239, 19, 458, 392, 294, 33, 160, 158, 133, 153, 144, 398, 385, 387, 359, 373, 374, 61, 73, 11, 267, 302, 270, 287, 321, 405, 17, 85, 91, 76, 80, 312, 272, 291, 319, 14, 88]

    processed_img, result = mp.process(img)
    pose_ret = mp.pose_unnormalization(result['pose'], img.shape)[0]["joints"]
    body_ret = pose_ret[body_idx]
    feet_ret = pose_ret[feet_idx]
    face_ret = mp.fmesh_unnormalization(result['fmesh'], img.shape)[0]["joints"][face_idx]
    hand_ret = mp.hand_unnormalization(result['hand'], img.shape)
    left_hand = hand_ret["Left"] if "Left" in hand_ret.keys() else np.zeros((21, 2))
    right_hand = hand_ret["Right"] if "Right" in hand_ret.keys() else np.zeros((21, 2))
    media_output = np.concatenate([
        body_ret[:, :2], feet_ret[:, :2], face_ret, right_hand, left_hand
    ], axis=0)
    return media_output

def load_mmpose_model(device_id):
    det_config_file = '../DWPose-onnx/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py'
    det_checkpoint_file = '../checkpoints/pose-related/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    detector = init_detector(det_config_file, det_checkpoint_file, device=device_id)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    config_file = '../DWPose-onnx/mmpose/configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-l_8xb32-270e_coco-wholebody-384x288.py'
    checkpoint_file = '../checkpoints/pose-related/dw-ll_ucoco_384.pth'
    pose_model = init_model(config_file, checkpoint_file, device=device_id)
    return {
        'det': detector,
        'pose': pose_model,
    }

def mmpose_process(model, img):
    # bbox & keypoints
    # keypoints: 133 * 3, 17 body + 6 feet + 68 face + 42 hands
    # body [:17]
    # feet [17:23]
    # face [23:91]
    # hands [91:133]
    det_result = inference_detector(model['det'], img)
    pred_instance = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                   pred_instance.scores > 0.3)]
    bboxes = bboxes[nms(bboxes, 0.3), :4]
    results = inference_topdown(model['pose'], img, bboxes)
    return_list = list()
    for item in results:
        item = item.to_dict()
        for k, v in item.items():
            if isinstance(v, dict):
                for sk, sv in v.items():
                    if isinstance(sv, np.ndarray):
                        item[k][sk] = sv.astype(np.float16)
        return_list.append(item)
    return return_list

def process_single_video(input_list, dtype, estimator, save_dir):
    video_list, device = input_list
    if estimator == "mmpose":
        register_all_modules()
        pose_model = load_mmpose_model(device)
    elif estimator == "mediapipe":
        settings = {
            'hand': True, 'landmark': False,
            'mesh': True, 'body': True,
            'selfie': False, 'holistic': False,
        }
        pose_model = MediaPipe(settings, video_mode=True)

    for video_dir in tqdm(video_list):
        output_list = list()
        fname = video_dir.rsplit("/", 1)[1].split('.')[0]
        save_path = f"{save_dir}/{fname}.pkl"
        if os.path.isfile(save_path):
            tqdm.write(f"{video_dir} has been estimated, skip")
            continue
        else:
            tqdm.write(f"Start estimate {video_dir}")
            with open(save_path, 'wb') as f:
                pickle.dump(output_list, f)

        if os.path.isfile(video_dir):
            cap = cv2.VideoCapture(video_dir)
            while(cap.isOpened()):
                ret, frame = cap.read()
                if not ret:
                    break
                frame =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if estimator == "mmpose":
                    output_kps = mmpose_process(pose_model, frame)
                elif estimator == "mediapipe":
                    output_kps = mediapipe2coco(pose_model, frame)
                output_list.append(output_kps)
        else:
            img_list = sorted(glob.glob(f"{video_dir}/*.{dtype}"))
            for img_idx, img_path in enumerate(img_list):
                img = cv2.imread(img_path)
                if estimator == "mmpose":
                    output_kps = mmpose_process(pose_model, img)
                elif estimator == "mediapipe":
                    output_kps = mediapipe2coco(pose_model, img)
                output_list.append(output_kps[None])

        tqdm.write(video_dir)
        if estimator == "mediapipe":
            pose_model.reset()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(save_path, 'wb') as f:
            pickle.dump(output_list, f)

def run_mp_cmd(processes, process_func, process_args):
    with Pool(processes) as p:
        outputs = list(tqdm(p.imap(process_func, process_args), total=len(process_args)))
    return outputs

def divide_list(l, n):
    chunk_size = (len(l) + n - 1) // n
    chunks = [l[i:i+chunk_size] for i in range(0, len(l), chunk_size)]
    # chunks = [l[i::n] for i in range(n)]
    return chunks

def assign_labels(m, n):
    label_size = (m + n - 1) // n
    labels = [i // label_size for i in range(m)]
    return labels

def load_error_id():
    dir_ = "/home/jiaopeiqi/data/MM_WLAuslan/train/rgb/"
    train_error_id = [24872, 49892, 74801, 99978]
    video_list = [os.path.join(dir_, f'{item}_kf_rgb.mp4') for item in train_error_id]

    for item in train_error_id:
        os.system(f'rm /home/jiaopeiqi/data/MM_WLAuslan/train/2d_skeleton/{item}_kf_rgb.pkl')

    return video_list

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    all_needed = [
        # '/home/jiaopeiqi/data/MM_WLAuslan/test/InTheWild/kl/rgb/*',
        # '/home/jiaopeiqi/data/MM_WLAuslan/test/InTheWild/kr/rgb/*',
        # '/home/jiaopeiqi/data/MM_WLAuslan/test/Studio/kl/rgb/*',
        '/home/jiaopeiqi/data/MM_WLAuslan/test/Studio/kr/rgb/*',
        '/home/jiaopeiqi/data/MM_WLAuslan/test/SyntheticBackground/kl/rgb/*',
        '/home/jiaopeiqi/data/MM_WLAuslan/test/SyntheticBackground/kr/rgb/*',
        '/home/jiaopeiqi/data/MM_WLAuslan/test/TemporalDisturb/kl/rgb/*',
        '/home/jiaopeiqi/data/MM_WLAuslan/test/TemporalDisturb/kr/rgb/*',
    ]
    for item in all_needed:
        dir_regular = item
        estimator = "mmpose"
        # estimator = "mediapipe"
        # output_dir = f"/home/jiaopeiqi/data/MM_WLAuslan/train/2d_skeleton/"
        output_dir = item.replace('rgb', '2d_skeleton')[:-2]
        os.makedirs(output_dir, exist_ok=True)
        ftype = "png"
        nprocessers = 10
        deviceIDs = GPUtil.getAvailable(
            order = 'first', limit = 4, maxLoad = 0.5, maxMemory = 0.5,
            includeNan=False, excludeID=[], excludeUUID=[]
        )
        cuda_ids = assign_labels(nprocessers, len(deviceIDs))
        cuda_ids = [f"cuda:{i}" for i in cuda_ids]

        dir_list = sorted(glob.glob(f"{dir_regular}"))[::-1]
        print(len(dir_list))
        nlist = divide_list(dir_list, nprocessers)

        # Debug
        # dir_list = load_error_id()
        # print(dir_list)
        # process_single_video(
        #     (dir_list, "cuda:0"), dtype=ftype, estimator=estimator, save_dir=output_dir
        # )
        # real run
        run_mp_cmd(
            nprocessers,
            partial(process_single_video, dtype=ftype, estimator=estimator, save_dir=output_dir),
            [*zip(nlist, cuda_ids)]
        )
        
        torch.cuda.empty_cache()
        time.sleep(60)
        torch.cuda.empty_cache()