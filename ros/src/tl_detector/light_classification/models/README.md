# Traffic-Light Classification by tensorflow models

The steps provided here are taken from [here](https://github.com/alex-lechner/Traffic-Light-Classification).

I added the summary of my actual steps as below.

## Environments
* OS: Unbuntu 16.04TS
* Python: 2.7
* Tensorflow: 1.4

### Install dependencies
- `pip install tensorflow-gpu==1.4` 
- `sudo apt-get install protobuf-compiler python-pil python-lxml python-tk`
- `git clone https://github.com/tensorflow/models.git`
- Navigate to the `models` and execute `git checkout f7e99c0`.
- Navigate to the `research` folder and execute `protoc object_detection/protos/*.proto --python_out=.`
- Execute ``export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim`` 
- For validating, Execute `python object_detection/builders/model_builder_test.py`

## Training

### Get the model

[SSD Inception V2 Coco (11/06/2017)](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_11_06_2017.tar.gz) 


### Get the dataset

[Drive location](https://drive.google.com/file/d/0B-Eiyn-CUQtxdUZWMkFfQzdObUE/view?usp=sharing)

### Get the models

Do `git clone https://github.com/tensorflow/models.git` inside the tensorflow directory


### Location of pre-trained models:
[pre-trained models zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

Download the required model tar.gz files and untar them into `/tensorflow/models/research/` directory with `tar -xvzf name_of_tar_file`.

### Creating TFRecord files:

`python data_conversion_udacity_sim.py --output_path sim_data.record`
`python data_conversion_udacity_real.py --output_path real_data.record`

---

---

## Using Inception SSD v2

### For Simulator Data

#### Training

`python object_detection/train.py --pipeline_config_path=config/ssd_inception-traffic-udacity_sim.config --train_dir=data/sim_training_data/sim_data_capture`

#### Saving for Inference

`python object_detection/export_inference_graph.py --pipeline_config_path=config/ssd_inception-traffic-udacity_sim.config --trained_checkpoint_prefix=data/sim_training_data/sim_data_capture/model.ckpt-10000 --output_directory=frozen_models/frozen_sim_inception/`


### For Real Data

#### Training

`python object_detection/train.py --pipeline_config_path=config/ssd_inception-traffic_udacity_real.config --train_dir=data/real_training_data`

#### Saving for Inference

`python object_detection/export_inference_graph.py --pipeline_config_path=config/ssd_inception-traffic_udacity_real.config --trained_checkpoint_prefix=data/real_training_data/model.ckpt-10000 --output_directory=frozen_models/frozen_real_inception/`

---
