# PPYOLOE on MMDetection

[**Original implementation on PaddleDetection.**](https://github.com/PaddlePaddle/PaddleDetection/tree/v2.4.0/configs/ppyoloe)

# Installation
Install mmdetection first if you haven't installed it yet. [Refer to mmdetection for details.](https://github.com/open-mmlab/mmdetection)

Tested mmdetection 2.25.2, other versions should work.

# Detail
The model is trained on V100*8.

# Datasets
```shell
mkdir data
ln -s path/to/coco data/
```

# Performance
| Model | Lr schd | box AP (val) | Download |
|:--------:|:-------:|:------:|:------:|
|PP-YOLOE-s  |    300e   |  43.0  | coming soon

# License
[Apache 2.0 license](LICENSE)

# Citation
```shell
@misc{ppdet2019,
title={PaddleDetection, Object detection and instance segmentation toolkit based on PaddlePaddle.},
author={PaddlePaddle Authors},
howpublished = {\url{https://github.com/PaddlePaddle/PaddleDetection}},
year={2019}
}

@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```