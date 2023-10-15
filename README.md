# [ICCV23] How Far Pre-trained Models Are from Neural Collapse on the Target Dataset Informs their Transferability
[Paper Link](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_How_Far_Pre-trained_Models_Are_from_Neural_Collapse_on_the_ICCV_2023_paper.pdf)

## Setup
```
Pytorch == 1.11.0
TorchVision == 0.12.0
```

## How-to
**Step 1** (Optional if you are using the same checkpoints as specified in models/group1/xx models)
```
python finetune_group1.py -m resnet50 -d sun397
```

**Step 2** Feature extraction for all candidate models on the SUN397 dataset. See `get_data()` function in `get_dataloader.py` file for all supported data sources.
```
python forward_feature_group1.py -d sun397
```

**Step 3** Evaluate the transferability of models 
```
python evaluate_metric_group1_cpu.py -me ncti -d sun397
```

**Step 4** Calculate the ranking correlation
```
python tw_group1.py --me ncti -d sun397
```

## Citation
If you find our work useful to your research, please cite
```
@inproceedings{wang2023far,
  title={How Far Pre-trained Models Are from Neural Collapse on the Target Dataset Informs their Transferability},
  author={Wang, Zijian and Luo, Yadan and Zheng, Liang and Huang, Zi and Baktashmotlagh, Mahsa},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5549--5558},
  year={2023}
}
```

## Acknowledgement:
This code repository is developed based on [SFDA](https://github.com/TencentARC/SFDA). 
