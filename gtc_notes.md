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

# Time series processing on Electronic Health Records: 
## Goals
* Discuss typical tools for RNN/LSTM
* Intro RNN, LSTMs, Keras, etc
* Hands on access.
* Be able to set up RNN, adapt to use case

## RNNs, LSTMs
* Similar to traditional feed-forward net, includes previous output states.
* limited to look only a few steps back (strictly).
* Useful for natural language project.
* Inputs are exponentially down weighted.
* Good visual in wildml's website's introduction-to-rnns

### LSTMs specifically.
* No vanishing gradient 
* can learn very large memory.
* each time step, measurement recorded and used as input to get prob(survival) prediction
* check colah.github.io 2015-08 lecture.

### KERAS
* Theano good for RNNs,python.
* Keras is useful modular python library.
	* Pandas is good for fast, efficient dataframe obj for data manipulation.
	* Numpy (scipy)
	* matplotlib

### On EHRs
* Contain medical treatments, history of patients over time.
* Irregular time series.
* HDF 5 data type.
	- Stores large amounts of sci data.
	- API supports most languages.
	- Multi OS workable.
	- binary format.
	- Efficient storage size.
	- Scales to large operational projects
* Measurments:
	- Statistics (gender,age, weight)
	- Vitals (heart rate)
	- Lab results (glucose, etc)
	- Interventions.
	- Drugging (dopamine, epinephrine)
* Not all measurements were taken for all patients.
* Dependent (alive vs not alive)

## Lab structure.
* Import libs
* Data prepration
	- Review
	- Normalization
	- Filling gaps
	- Data sequencing
* Architect LSTM
* Build model
* Eval
* Visualize results
* Compare to PRISM3, PIM2

### Import libs
* Pretty standard.

### Data preperatiaon
* Can review in plot
* Some things specific to the datatype, but pandas is good to import into python.
* Normalize data ( take mean out, divide over std dev).
* Fill gaps
	- No standard option for filling gaps.
	- currently forward filling, zeropad the left side of the signal. (choose what works with your data)
* Pad every encounter per patient so that each encounter has the same number of datapoints.
	- Zero pad.
	- Can do with theano.

### Architect LSTM