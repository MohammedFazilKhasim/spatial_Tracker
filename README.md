

# SpatialTracker: Tracking Any 2D Pixels in 3D Space

## Requirements 

The inference code was tested on

* Ubuntu 20.04
* Python 3.10
* [PyTorch](https://pytorch.org/) 2.1.1
* 1 NVIDIA GPU (RTX A6000) with CUDA version 11.8. (Other GPUs are also suitable, and 22GB GPU memory is sufficient for dense tracking (~10k points) with our code.)
### Setup an environment
```shell
conda create -n SpaTrack python==3.10
conda activate SpaTrack
```
### Install PyTorch
```shell
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
```

### Other Dependencies
```shell
pip install -r requirements.txt
```
<mark>Note:</mark> Please follow the version of the dependencies in `requirements.txt` to avoid potential conflicts. 

## Depth Estimator
In our default setting, monocular depth estimator is needed to acquire the metric depths from video input. There are several models for options ([ZoeDepth](https://github.com/isl-org/ZoeDepth), [Metric3D](https://github.com/YvanYin/Metric3D), [UniDepth](https://github.com/lpiccinelli-eth/UniDepth) and [DepthAnything](https://github.com/LiheYoung/Depth-Anything)). 
We take ZoeDepth as default model. **Download** `dpt_beit_large_384.pt`, `ZoeD_M12_K.pt`, `ZoeD_M12_NK.pt` into `models/monoD/zoeDepth/ckpts`. 

## Data
Our method supports **`RGB`** or **`RGBD`** videos input. We provide the `checkpoints` and `example_data` at the  [Goolge Drive](https://drive.google.com/drive/folders/1UtzUJLPhJdUg2XvemXXz1oe6KUQKVjsZ?usp=sharing). Please download the `spaT_final.pth` and put it into `./checkpoints/`.  

### RGB Videos
For `example_data`, we provide the `butterfly.mp4` and `butterfly_mask.png` as an example. Download the `butterfly.mp4` and `butterfly_mask.png` into `./assets/`. And run the following command: 

```shell
python demo.py --model spatracker --downsample 1 --vid_name butterfly --len_track 1 --fps_vis 15  --fps 1 --grid_size 40 --gpu ${GPU_id}
```
### RGBD Videos
we provide the `sintel_bandage.mp4`, `sintel_bandage.png` and `sintel_bandage/` in `example_data`. `sintel_bandage/` includes the depth map of the `sintel_bandage.mp4`. Download the `sintel_bandage.mp4`, `sintel_bandage.png` and `sintel_bandage/` into `./assets/`. And run the following command: 
```shell
python demo.py --model spatracker --downsample 1 --vid_name sintel_bandage --len_track 1 --fps_vis 15  --fps 1 --grid_size 60 --gpu ${GPU_id} --point_size 1 --rgbd # --vis_support (optional to visualize all the points)
```

## Visualization 3D Trajectories
Firstly, please make sure that you have installed [blender](https://www.blender.org/).
We provide the visualization code for blender: 
```shell
/Applications/Blender.app/Contents/MacOS/Blender -P create.py -- --input ./vis_results/sintel_bandage_3d.npy
```
For example, the `sintel_bandage` looked like 
![](assets/sintel.gif)

## Citation
If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{SpatialTracker,
    title={SpatialTracker: Tracking Any 2D Pixels in 3D Space},
    author={Xiao, Yuxi and Wang, Qianqian and Zhang, Shangzhan and Xue, Nan and Peng, Sida and Shen, Yujun and Zhou, Xiaowei},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2024}
}
```

## Acknowledgement
Spatialtracker is built on top of [Cotracker](co-tracker.github.io) codebase. We appreciate the authors for their greate work and follow the License of Cotracker.
