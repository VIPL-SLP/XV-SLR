import os, pickle, pdb, cv2, json
import numpy as np
from tqdm import tqdm

error_id = dict()

for scene in ['STU', 'SYN', 'TED']:
    for view in ['kl', 'kr']:

        tar_dir = f'/home/jiaopeiqi/data/MM_WLAuslan/test/{scene}/{view}/2d_skeleton'
        video_dir = f'/home/jiaopeiqi/data/MM_WLAuslan/test/{scene}/{view}/rgb'
        video_list = os.listdir(tar_dir)
        video_list = list(filter(lambda x: x.endswith('pkl'), video_list))
        print(len(video_list))

        for item in tqdm(video_list):
            file = pickle.load(open(os.path.join(tar_dir, item), 'rb'))
            filtered_skeleton = []
            for frame in file:
                if len(frame) == 1:
                    needed = frame[0]
                else:
                    det_score = [float(x['pred_instances']['bbox_scores']) for x in frame]
                    max_score = max(det_score)
                    needed = frame[det_score.index(max_score)]
                    # pdb.set_trace()
                kps = needed['pred_instances']['keypoints']
                kps_score = needed['pred_instances']['keypoint_scores']
                filtered_skeleton.append(np.concatenate([kps[0], kps_score[0][:,None]], axis=-1))
            if len(filtered_skeleton) == 0:
                if not f'{scene}_{view}' in error_id:
                    error_id[f'{scene}_{view}'] = []
                error_id[f'{scene}_{view}'].append(item.split('_')[0])
                continue
            # check for video frames
            v_path = os.path.join(video_dir, item.replace('.pkl', '.mp4'))
            # pdb.set_trace()
            cap = cv2.VideoCapture(v_path)
            if not cap.isOpened():
                print(f"Can not open video: {v_path}")
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not len(filtered_skeleton) == frames:
                print(f'error id {item.split("_")[0]}')
                error_id.append(item.split("_")[0])
                continue
            del cap
            
            filtered_skeleton = np.stack(filtered_skeleton, axis=0)
            fname = item.rsplit('.', 1)[0]
            np.save(f'{tar_dir}/{fname}.npy', filtered_skeleton)
    # pdb.set_trace()
print(error_id)
json.dump(error_id, open('test_error_id.json'), indent=4)