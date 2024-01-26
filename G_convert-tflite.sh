## CHANGE BASED ON THE DIRECTORIES NEEDED ##

mkdir custom_model_lite
python models/research/object_detection/export_tflite_graph_tf2.py --trained_checkpoint_dir=training --output_directory=custom_model_lite  --pipeline_config_path=models/mymodel/pipeline_file.config

python 3_tflite-converter.py

cp labelmap.txt custom_model_lite
cp labelmap.pbtxt custom_model_lite
cp models/mymodel/pipeline_file.config custom_model_lite

cd custom_model_lite
curl https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/TFLite_detection_image.py --output TFLite_detection_image.py
curl https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/TFLite_detection_video.py --output TFLite_detection_video.py
curl https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/TFLite_detection_webcam.py --output TFLite_detection_webcam.py
curl https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/TFLite_detection_stream.py --output TFLite_detection_stream.py

cd ../