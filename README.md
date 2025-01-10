# Basic Information
This is the fusion stage training and inference of the VIPL-SLP submission for [CV-ISLR challenge](https://uq-cvlab.github.io/MM-WLAuslan-Dataset/docs/en/www).

It worth noting that current version is unclean and needs a significant re-orgainzed. 

# Data Preparation
Download the estimated keypoints, extracted RGB and depth features from alipan, and put them under the root of project:
- estimated keypoints: https://www.alipan.com/t/q5X0dN233lDMVAasaGyO
- extracted RGB features: https://www.alipan.com/t/T42Rgvw0VtlaggJji7tM
- extracted depth features: https://www.alipan.com/t/OI0nmWKwj71pE7epRx9e

Feel free to contact us if the link is invalid.

# Environment Configuration
```bash
conda env create -f environment.yml
```

# Evaluation
[to do] Download the pretrained weights and put them in the ./weights


## RGB Track
- For RGB data:
```python
python main.py --config ./configs/test_single_rgb.yaml --load-weights weights/single_rgb.pt
```
Expected performance: Average Topk-1 : 34.55%

- For skeleton data:
```python
python main.py --config ./configs/test_single_skeleton.yaml --load-weights weights/sk_phase2.pt
```
Expected performance: Average Topk-1 : 46.00%

- For RGB+Skeleton data:
```python
python main.py --config ./configs/test_fusion_rgbd.yaml --load-weights ./weights/fusion_rgbd.pt
```
Expected performance: Average Topk-1 : 56.87%

## RGB-D Track
- For Depth data:
```python
python main.py --config ./configs/test_single_depth.yaml --load-weights ./weights/single_depth.pt
```
Expected performance: Average Topk-1 : 28.84%

- For RGB+Skeleton+Depth data:
```python
python main.py --config ./configs/test_fusion_rgbd.yaml --load-weights ./weights/fusion_rgbd.pt
```
Expected performance: Average Topk-1 : 57.98%

# Training
## RGB Track
```python
python main.py --config ./configs/train_fusion_rgb.yaml
```

## RGB-D Track
```python
python main.py --config ./configs/train_fusion_rgbd.yaml
```