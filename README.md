# DeSTSeg

Official PyTorch implementation of [DeSTSeg]([https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_DeSTSeg_Segmentation_Guided_Denoising_Student-Teacher_for_Anomaly_Detection_CVPR_2023_paper.html]) - CVPR 2023

## Datasets

We use the MVTec AD dataset for experiments. To simulate anomalous image, the Describable Textures Dataset (DTD) is also adopted in our work. Users can run the **download_dataset.sh** script to download them directly.

```
./scripts/download_dataset.sh
```

## Installation

Please install the dependency packages using the following command by **pip**:

```
pip install -r requirements.txt
```

## Training and Testing

To get started, users can run the following command to train the model on all categories of MVTec AD dataset:

```
python train.py --gpu_id 0 --num_workers 16
```

Users can also customize some default training parameters by resetting arguments like `--bs`, `--lr_DeST`, `--lr_res`, `--lr_seghead`, `--steps`, `--DeST_steps`, `--eval_per_steps`, `--log_per_steps`, `--gamma` and `--T`.

To specify the training categories and the corresponding data augmentation strategies, please add the argument `--custom_training_category` and then add the categories after the arguments `--no_rotation_category`, `--slight_rotation_category` and `--rotation_category`. For example, to train the `screw` category and the `tile` category with no data augmentation strategy, just run the following command:

```
python train.py --gpu_id 0 --num_workers 16 --custom_training_category --no_rotation_category screw tile
```

To test the performance of the model, users can run the following command:

```
python eval.py --gpu_id 0 --num_workers 16
```

## Pretrained Checkpoints

Download pretrained checkpoints [here](https://www.icloud.com.cn/iclouddrive/051C6C9EWaC9e6XnLqtmghX0A#saved%5Fmodel) and put the checkpoints under `<project_dir>/saved_model/`.

## Citation

```
@inproceedings{zhang2023destseg,
  title={DeSTSeg: Segmentation Guided Denoising Student-Teacher for Anomaly Detection},
  author={Zhang, Xuan and Li, Shiyu and Li, Xi and Huang, Ping and Shan, Jiulong and Chen, Ting},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3914--3923},
  year={2023}
}
```
