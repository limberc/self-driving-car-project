# Mask RCNN Object Detection

In this work, we first apply Mask RCNN pre-trained model to take a warm up. 

After familiar to the data and techniques, we would like to build a pipeline to make it easier to apply to any project. We would like to train the model first in MS COCO dataset 2017(new release). 

## Configurations

We'll be using a model trained on the [MS-COCO dataset](http://cocodataset.org/#download). The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.

For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.

**Reference**

1. He K, Gkioxari G, Dollár P, et al. Mask r-cnn[C]//Computer Vision (ICCV), 2017 IEEE International Conference on. IEEE, 2017: 2980-2988.