# BirdBotWindows
Files to run BirdBot on Windows 10 with CPU or GPU. With the current build I would recommend installing TensorFlow GPU 1.15.

### HELPFUL YOUTUBE VIDEOS TO RE-TRAIN THE BIRDMODEL ###

1.) How to Install TensorFlow Object Detection in 2020 (Webcam and Images!)
https://www.youtube.com/watch?v=usR2LQuxhL4

2.) How to Create a Custom Object Detector with TensorFlow in 2020
https://www.youtube.com/watch?v=C5-SEZ_IvaM

3.) How to Train a Custom Model for Object Detection (Local and Google Colab!) <- We are doing it locally
https://www.youtube.com/watch?v=_gGI91BmIdk

### SETING UP CONDA ENVIORNMENT ###

o https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe

o pip install TensorFlow==1.15 lxml pillow matplotlib jupyter contextlib2 cython tf_slim

o git clone https://github.com/tensorflow/models.git

o protoc object_detection/protos/*.proto --python_out=.

o python setup.py build

o python setup.py install

o pip install pycocotools (or) pip install git+https://github.com/philferriere/cocoa...^&subdirectory=PythonAPI

### COMMANDS TO RUN BIRDBOTML ###

o cd BirdBotWindows

o python BirdBotML.py

### COMMANDS TO TRAIN BIRDBOTML - IMAGES FOLDER REQ. ###

o cd BirdBotWindows

o python partition_dataset.py

o python xml_to_csv.py

o python generate_tfrecord.py --csv_input=B:/BirdBot/BirdModelTraining/data/train_labels.csv --image_dir=B:/BirdBot/BirdModelTraining/images  --output_path=B:/BirdBot/BirdModelTraining/data/train.record

o python generate_tfrecord.py --csv_input=B:/BirdBot/BirdModelTraining/data/test_labels.csv --image_dir=B:/BirdBot/BirdModelTraining/images --output_path=B:/BirdBot/BirdModelTraining/data/test.record

o python train.py --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_fpn.config --logtostderr

o python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_mobilenet_v1_fpn.config --trained_checkpoint_prefix training/model.ckpt-50000 --output_directory BirdModel

###SUCCESS - YOU CAN NOW RUN YOUR CUSTOM TENSORFLOW MODEL WITH COMMAND BELOW - SUCCESS###

o python BirdBotML.py

###EXPERIMENTAL - EVERYTHING BELOW IS FOR 8-bit QUANTIZATION WHICH IS REQUIRED FOR TPU HARDWARE - EXPERIMENTAL###

o python export_tflite_ssd_graph.py --pipeline_config_path=training/ssd_mobilenet_v1_fpn.config --trained_checkpoint_prefix=training/model.ckpt-30000 --output_directory=BirdModelTFlite --add_postprocessing_op=true

o tflite_convert --output_file=BirdModelTFlite/detect.tflite --graph_def_file=B:/BirdBot/BirdModel/models/research/object_detection/BirdModelTFlite/tflite_graph.pb --input_arrays=normalized_input_image_tensor --output_arrays=TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3 --input_shape=1,480,480,3 --allow_custom_ops --optimizations={tf.lite.Optimize.DEFAULT} --inference_type=QUANTIZED_UINT8 --mean_values=128 --std_dev_values=128 --default_ranges_min=0 --default_ranges_max=10

### FIGURE OUT --mean_values=128 --std_dev_values=128 --default_ranges_min=0 --default_ranges_max=10 ###
### TPU models are jank AF until I figure out how to properly convert/train my models in 8-bit quantization - doesn't run fast ###

o launch "ubuntu" in conda env

o go to C:\Users\Tyler\AppData\Local\Packages\CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc\LocalState\rootfs

o update detect.tflite

o sudo edgetpu_compiler detect.tflite

o python RunTPUExample.py
