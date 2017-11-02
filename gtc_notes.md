#Full Motion video Datasets:

#Detection + tracking lab:
## To get from it.
* Format FMV data + annotations,
* Set up obj detect pipeline in TF
* Get used to TF
* Quant assessment of performance.
* CLEAR MOT (class Events, activities, relationships multi obj tracking) tracking metrics
* Focus on low/med alt platforms.

## Dataset
* VIVID dataset (video verification of ID)
* 4 vis band datasets, 30 frames per second, 640x480,1 min

## Basic basics (should already know)
* Prepare Inputs. numpy / TFRecords (for large datasets).
* Build graph, create inference, loss, training nodes.
* Train
* etc.
* will look at Tensorboard.

## Begin
* Uses tensorflow object detection API.
* Algorithms implemented:
	* Single Shot Multibox Detector (SSD) with MobileNets
		* https://arxiv.org/abs/1512.02325
	* SSD with Inception v2
	* Region-Based Fully Convolutional Networks (R-FCN) with Resnet 101
	* Faster RCNN with Resnet 101
	* Faster RCNN with Inception Resnet v2

* second, last options used.
	* SSDs can get near common FMV rates, can produce many false negs, false pos
	* Use TensorRT to optimize, deepstream for system scaling.

* Use sloth to annotate ground truth.
	* the lab focuses on obj detection, not segmentation.
	* annot. frame rate 1/10 of vid rate, detects better. In JSON.
	* Convert to xml, use PASCAL VOC format
		* pandas to data storage better.
* From this txt xml file, use COCO config file. to explicitly list object types with numeric id.
	* FORMAT:
	> item {
  	>	name: 'vehicle'
  	>	id: 1
  	>	display_name: 'vehicle'
	> }

* Interference graph (optimized model) used to detect objects, before we look at actual data.
	* Review model trained on COCO data fed VIVID data

* Model configuration for Tensorflow Obj detect is in config files, will save a copy to include.

* Evaluate accuracy by 
	* area of intersection (overlap)/area of union (total area)
		* Good for intuition, but doesn't cover temporal changes.
	* CLEAR MOT
		* "The CLEAR 2006 evaluation."
		* 1 function is multiple object tracking precision (MOTP)
			* Total error in estimated position.
		* other is multiple object tracking accuracy (MOTA)
			* Sum of misses, false positives, mismatches / ground truth
* sort  is python package with traditional tracking functions (KF and such)