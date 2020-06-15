Detectron2 is Facebook AI Research's next generation software system
that implements state-of-the-art object detection algorithms.

## This project to count products accumulation in video stream 

## Installation

See [INSTALL.md](INSTALL.md).\
And clone the repository ```https://github.com/nguyentienanh2303/count_product_on_video.git```

## How to run

Clone the repository and change directory to repository
1. Pick a model and its config file from model zoo, for example, faster_rcnn_R_50_C4_3x.yaml
``` cd demo/\
python product_counter.py --config-file ../configs/COCO-Detection/faster_rcnn_R_50_C4_3x.yaml\
--video-input path/to/your/video\
--output path/to/save/result\
--opts MODEL.WEIGHTS path/to/model_weight
```
Download following model weight: https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_C4_3x/137849393/model_final_f97cb7.pkl from model zoo with config file respectively.

## License

Detectron2 is released under the [Apache 2.0 license](LICENSE).

## Citing Detectron2

If you use Detectron2 in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```
