git clone https://github.com/Cartucho/mAP 
cd mAP
rm input/detection-results/*
rm input/ground-truth/*
rm input/images-optional/*
wget https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/util_scripts/calculate_map_cartucho.py

cd ../
cp images/test/* mAP/input/images-optional # Copy images and xml files
mv mAP/input/images-optional/*.xml mAP/input/ground-truth  # Move xml files to the appropriate folder
python mAP/scripts/extra/convert_gt_xml.py

python 4_map.py

cd mAP
cd ../
cp labelmap.txt mAP
cd mAP
python calculate_map_cartucho.py --labels=labelmap.txt
cd ../