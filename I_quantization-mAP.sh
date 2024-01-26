rm mAP/input/detection-results/*
python 5_quantization-mAP.py
cd mAP
python calculate_map_cartucho.py --labels=labelmap.txt
cd ../