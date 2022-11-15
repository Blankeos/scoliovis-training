# ScolioVis Training

> ⚠ Documentation is still under construction! Be careful.

This repository contains instructions for replicating our training process for [ScolioVis].

We also store the training scripts for evaluating the **KeypointRCNN Model**, as well as the instructions for replicating the training process for our research on Google Colab.

### Instructions

1. Request [Dataset 16: 609 spinal anterior-posterior x-ray images](http://spineweb.digitalimaginggroup.ca/Index.php?n=Main.Datasets#Dataset_16.3A_609_spinal_anterior-posterior_x-ray_images) from SpineWeb

   You should receive a file called `boostnet_labeldata.zip` from them. Unzip that and save the root of the `boostnet_labeldata` folder on your Google Drive.

2. Execute and follow the steps on our **[Preprocessing Notebook]**

   The final data should look like:

   ```
   keypointsrcnn_data
   ├─images
   | ├─train
   | | ├─img_1.jpg
   | | └─(..).jpg
   | └─val
   |   ├─img.jpg
   |   └─(..).jpg
   └─labels
   | ├─img_1.json
   | └─(..).json
   └─val
       ├─img.json
       └─(..).json
   ```

   After following the instructions, your final data should be saved on your Google Drive for training later on.

3. Execute and follow the steps on our **[Training Notebook]**

4. You finished everything! 🎉

### References:

- Training Scripts:

  - The training scripts in this repository can be obtained from either [**pytorch**/vision/references/detection](https://github.com/pytorch/vision/tree/main/references/detection) or [**alexppppp**/keypoint_rcnn_training_pytorch](https://github.com/alexppppp/keypoint_rcnn_training_pytorch).
  - I then changed `coco_eval.py` to contain the following since the model has 4 points for each instance detected:
    ```py
    # some code here
    ```

- [Detailed Medium Article on Custom Keypoint RCNN Training](https://medium.com/@alexppppp/how-to-train-a-custom-keypoint-detection-model-with-pytorch-d9af90e111da)

> **pytorch**/vision/references/detection content below ⬇

---

## Object detection reference training scripts

This folder contains reference training scripts for object detection.
They serve as a log of how to train specific models, to provide baseline
training and evaluation scripts to quickly bootstrap research.

To execute the example commands below you must install the following:

```
cython
pycocotools
matplotlib
```

You must modify the following flags:

`--data-path=/path/to/coco/dataset`

`--nproc_per_node=<number_of_gpus_available>`

Except otherwise noted, all models have been trained on 8x V100 GPUs.

### Faster R-CNN ResNet-50 FPN

```
torchrun --nproc_per_node=8 train.py\
    --dataset coco --model fasterrcnn_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3 --weights-backbone ResNet50_Weights.IMAGENET1K_V1
```

### Faster R-CNN MobileNetV3-Large FPN

```
torchrun --nproc_per_node=8 train.py\
    --dataset coco --model fasterrcnn_mobilenet_v3_large_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3 --weights-backbone MobileNet_V3_Large_Weights.IMAGENET1K_V1
```

### Faster R-CNN MobileNetV3-Large 320 FPN

```
torchrun --nproc_per_node=8 train.py\
    --dataset coco --model fasterrcnn_mobilenet_v3_large_320_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3 --weights-backbone MobileNet_V3_Large_Weights.IMAGENET1K_V1
```

### FCOS ResNet-50 FPN

```
torchrun --nproc_per_node=8 train.py\
    --dataset coco --model fcos_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3  --lr 0.01 --amp --weights-backbone ResNet50_Weights.IMAGENET1K_V1
```

### RetinaNet

```
torchrun --nproc_per_node=8 train.py\
    --dataset coco --model retinanet_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3 --lr 0.01 --weights-backbone ResNet50_Weights.IMAGENET1K_V1
```

### SSD300 VGG16

```
torchrun --nproc_per_node=8 train.py\
    --dataset coco --model ssd300_vgg16 --epochs 120\
    --lr-steps 80 110 --aspect-ratio-group-factor 3 --lr 0.002 --batch-size 4\
    --weight-decay 0.0005 --data-augmentation ssd --weights-backbone VGG16_Weights.IMAGENET1K_FEATURES
```

### SSDlite320 MobileNetV3-Large

```
torchrun --nproc_per_node=8 train.py\
    --dataset coco --model ssdlite320_mobilenet_v3_large --epochs 660\
    --aspect-ratio-group-factor 3 --lr-scheduler cosineannealinglr --lr 0.15 --batch-size 24\
    --weight-decay 0.00004 --data-augmentation ssdlite
```

### Mask R-CNN

```
torchrun --nproc_per_node=8 train.py\
    --dataset coco --model maskrcnn_resnet50_fpn --epochs 26\
    --lr-steps 16 22 --aspect-ratio-group-factor 3 --weights-backbone ResNet50_Weights.IMAGENET1K_V1
```

### Keypoint R-CNN

```
torchrun --nproc_per_node=8 train.py\
    --dataset coco_kp --model keypointrcnn_resnet50_fpn --epochs 46\
    --lr-steps 36 43 --aspect-ratio-group-factor 3 --weights-backbone ResNet50_Weights.IMAGENET1K_V1
```
