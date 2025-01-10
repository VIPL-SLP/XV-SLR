import json, os, cv2, pdb
from tqdm import tqdm

def resize_img(img, dsize=(256,256)):
    img = cv2.resize(img, dsize, interpolation=cv2.INTER_LANCZOS4)
    return img

def generate_gloss_dict():

    original_dict = json.load(open('../data/Dictionary.json', 'r'))
    new_dict = {
        'gloss2id': {},
        'id2gloss': {}
    }
    count = 0
    for k, v in original_dict.items():
        new_dict['gloss2id'][k] = count
        new_dict['id2gloss'][count] = k
        count += 1
    print(f'total {count} glosses')
    json.dump(new_dict, open('../data/gloss_dict.json', 'w'), indent=4)

def extract_single_video(args):
    path, save_dir = args
    cap = cv2.VideoCapture(path)
    os.makedirs(save_dir, exist_ok=True)
    index = 0

    while cap.isOpened():  
        ret, frame = cap.read()  
        if not ret:  
            break
        height, width = frame.shape[:2]
        assert height == 408 and width == 512
        # center crop
        frame = frame[:,52:512-52]
        # pdb.set_trace()
        frame = resize_img(frame)
        cv2.imwrite(os.path.join(save_dir, f'{str(index).zfill(6)}.jpg'), frame)
        index += 1

def extract_video_frames():
    from concurrent.futures import ThreadPoolExecutor
    all_args = []
    for scene in ['ITW', 'STU', 'SYN', 'TED']:
        for view in ['kl', 'kr']:
            tar_dir = f'/home/jiaopeiqi/data/MM_WLAuslan/test/{scene}/{view}/rgb'
            save_dir = f'/home/jiaopeiqi/data/MM_WLAuslan/test/{scene}/{view}/rgb_frames'
            for item in os.listdir(tar_dir):
                temp = []
                temp.append(os.path.join(tar_dir, item))
                temp.append(os.path.join(save_dir, item.split('_')[0]))
                all_args.append(temp)
    # pdb.set_trace()
    executor = ThreadPoolExecutor(16)
    for ret in executor.map(extract_single_video, all_args):
        pass

def move_rename_check():
    for scene in ['ITW', 'STU', 'SYN', 'TED']:
        for view in ['kl', 'kr']:
            tar_dir = f'/home/jiaopeiqi/data/MM_WLAuslan/test/{scene}/{view}/rgb'
            save_dir = f'/home/jiaopeiqi/data/MM_WLAuslan/test/{scene}/{view}/rgb_frames'
            for item in tqdm(os.listdir(tar_dir)):
                fid = item.split('_')[0]
                cap = cv2.VideoCapture(os.path.join(tar_dir, item))
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                frames2 = len(os.listdir(os.path.join(save_dir, fid)))
                assert frames == frames2
            os.system(f'rm -rf {tar_dir}')
            os.system(f'mv {save_dir} {tar_dir}')

def check_all():
    for mode in ['train', 'val']:
        tar_dir = f'/home/jiaopeiqi/data/MM_WLAuslan/{mode}/kf/rgb_videos'
        save_dir = f'/home/jiaopeiqi/data/MM_WLAuslan/{mode}/kf/rgb'
        for item in tqdm(os.listdir(tar_dir)):
            fid = item.split('_')[0]
            cap = cv2.VideoCapture(os.path.join(tar_dir, item))
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frames2 = len(os.listdir(os.path.join(save_dir, fid)))
            assert frames == frames2

def generate_test_info():
    tar_dir = '/home/jiaopeiqi/data/MM_WLAuslan/test/InTheWild/kinect_l/rgb'
    test_info = {}
    for item in os.listdir(tar_dir):
        fid = item.split('_')[0]
        test_info[fid] = 'none'
    print(len(test_info))
    json.dump(test_info, open('../data/test.json', 'w'), indent=4)

def generate_info_file():

    for mode in ['train', 'val']:
        origianl_file = json.load(open(f'../data/original_info/{mode}.json', 'r'))
        new_file = []
        for k, v in origianl_file.items():
            item = {
                'fid': k,
                'label': v,
                'view': 'kf',
                'save_path': f'./MM_WLAuslan/{mode}/'
            }
            new_file.append(item)
        json.dump(new_file, open(f'../data/{mode}_self.json', 'w'), indent=4)

    for mode in ['Test']:
        for scene in ['ITW', 'STU', 'SYN', 'TED']:
            origianl_file = json.load(open(f'../data/original_info/{mode}_{scene}.json', 'r'))
            new_file = []
            for k, v in origianl_file.items():
                item = {
                    'fid': k,
                    'label': v,
                    'view': 'kf',
                    'save_path': f'./MM_WLAuslan/{mode.lower()}/{scene}/'
                }
                new_file.append(item)
            json.dump(new_file, open(f'../data/{mode.lower()}_{scene}_self.json', 'w'), indent=4)

def view_skeleton():

    def draw_skeleton(frame, kps):
        for i in range(len(kps)):
            x, y = int(kps[i][0]), int(kps[i][1])
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
        return frame

    import numpy as np
    input_video_path = '/home/jiaopeiqi/data/MM_WLAuslan/train/kf/rgb_videos/87284_kf_rgb.mp4'
    cap = cv2.VideoCapture(input_video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))  
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter('view_skeleton_2.mp4', fourcc, fps, (frame_width, frame_height))

    skeleton = np.load('/home/jiaopeiqi/data/MM_WLAuslan/train/kf/2d_skeleton/87284_kf_rgb.npy')
    index = 0

    while cap.isOpened():  
        ret, frame = cap.read()  
        if not ret:  
            break
        kps = skeleton[index]
        frame = draw_skeleton(frame, kps)
        out.write(frame)
        index += 1

def convert_caffe_weights(caffe_weight, output_path):
    import pickle
    import torch
    import re
    import sys

    c2_weights = caffe_weight # 'pretrained/i3d_baseline_32x2_IN_pretrain_400k.pkl'
    pth_weights_out = output_path # 'pretrained/i3d_r50_kinetics.pth'

    c2 = pickle.load(open(c2_weights, 'rb'), encoding='latin')['blobs']
    c2 = {k:v for k,v in c2.items() if 'momentum' not in k}

    downsample_pat = re.compile('res(.)_(.)_branch1_.*')
    conv_pat = re.compile('res(.)_(.)_branch2(.)_.*')
    nl_pat = re.compile('nonlocal_conv(.)_(.)_(.*)_.*')

    m2num = dict(zip('abc',[1,2,3]))
    suffix_dict = {'b':'bias', 'w':'weight', 's':'weight', 'rm':'running_mean', 'riv':'running_var'}

    key_map = {}
    key_map.update({'conv1.weight':'conv1_w',
                'bn1.weight':'res_conv1_bn_s',
                'bn1.bias':'res_conv1_bn_b',
                'bn1.running_mean':'res_conv1_bn_rm',
                'bn1.running_var':'res_conv1_bn_riv',
                'fc.weight':'pred_w',
                'fc.bias':'pred_b',
                })

    for key in c2:

        conv_match = conv_pat.match(key)
        if conv_match:
            layer, block, module = conv_match.groups()
            layer, block, module = int(layer), int(block), m2num[module]
            name = 'bn' if 'bn_' in key else 'conv'
            suffix = suffix_dict[key.split('_')[-1]]
            new_key = 'layer%d.%d.%s%d.%s'%(layer-1, block, name, module, suffix)
            key_map[new_key] = key

        ds_match = downsample_pat.match(key)
        if ds_match:
            layer, block = ds_match.groups()
            layer, block = int(layer), int(block)
            module = 0 if key[-1]=='w' else 1
            name = 'downsample'
            suffix = suffix_dict[key.split('_')[-1]]
            new_key = 'layer%d.%d.%s.%d.%s'%(layer-1, block, name, module, suffix)
            key_map[new_key] = key

        nl_match = nl_pat.match(key)
        if nl_match:
            layer, block, module = nl_match.groups()
            layer, block = int(layer), int(block)
            name = 'nl.%s'%module
            suffix = suffix_dict[key.split('_')[-1]]
            new_key = 'layer%d.%d.%s.%s'%(layer-1, block, name, suffix)
            key_map[new_key] = key

    import sys
    sys.path.append('..')

    from modules.visual_extractor import resnet
    pth = resnet.I3Res50(num_classes=400, use_nl=True)
    state_dict = pth.state_dict()

    new_state_dict = {key: torch.from_numpy(c2[key_map[key]]) for key in state_dict if key in key_map}
    torch.save(new_state_dict, pth_weights_out)
    # torch.save(key_map, pth_weights_out+'.keymap')

    # check if weight dimensions match
    for key in state_dict:

        if key not in key_map:
            continue

        c2_v, pth_v = c2[key_map[key]], state_dict[key]
        assert str(tuple(c2_v.shape))==str(tuple(pth_v.shape)), 'Size Mismatch'
        print ('{:23s} --> {:35s} | {:21s}'.format(key_map[key], key, str(tuple(c2_v.shape))))

def unzip():
    for scene in ['Studio', 'SyntheticBackground', 'TemporalDisturb']:
        for view in ['kinect_l', 'kinect_r']:
            save_dir = f'/home/jiaopeiqi/data/MM_WLAuslan/test/{scene}/{view}'
            os.makedirs(save_dir, exist_ok=True)
            for mode in ['depth', 'rgb']:
                tar_file = f'/home/myc/data/MM_WLAuslan/test/{scene}/{view}/{mode}.zip'
                cmd = f'unzip -o {tar_file} -d {save_dir}'
                os.system(cmd)

if __name__ == '__main__':
    # generate_test_info()
    # generate_info_file()
    # view_skeleton()
    # extract_video_frames()
    # check_all()
    # unzip()
    # generate_info_file()
    # convert_caffe_weights('../checkpoints/I3D/i3d_baseline_32x2_IN_pretrain_400k.pkl', '../checkpoints/I3D/i3d_r50_kinetics.pth')
    # convert_caffe_weights('../checkpoints/I3D/i3d_nonlocal_32x2_IN_pretrain_400k.pkl', '../checkpoints/I3D/i3d_r50_nl_kinetics.pth')
    # extract_video_frames()
    move_rename_check()