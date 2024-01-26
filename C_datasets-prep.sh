# PUT YOUR DATASETS IN DATASETS FOLDER ##
# THE DATASETS SHOULD UNDERGO DATA AUGMENTATION BEFORE PLACING IN THE MAIN FOLDER ##

mkdir images
cd images
mkdir train 
mkdir validation
mkdir test
mkdir test-quan
cd ../
cp -r datasets/* images/train  
cp -r datasets/* images/validation
cp -r datasets/* images/test    
cp -r datasets/* images/test-quan

## PUT THE RIGHT LABELS HERE BASED ON THE LABELLING  ##
## NOTE: CHECK FOR CAPS

cat <<EOF>> labelmap.txt
cirrus   
cumulus   
nimbus   
Stratus   
EOF

wget https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/util_scripts/create_csv.py
wget https://raw.githubusercontent.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/master/util_scripts/create_tfrecord.py

python create_csv.py
python create_tfrecord.py --csv_input=images/train_labels.csv --labelmap=labelmap.txt --image_dir=images/train --output_path=train.tfrecord
python create_tfrecord.py --csv_input=images/validation_labels.csv --labelmap=labelmap.txt --image_dir=images/validation --output_path=val.tfrecord