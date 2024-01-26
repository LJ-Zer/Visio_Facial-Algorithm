# mkdir protoc-3.12.4-win64-1
# cd protoc-3.12.4-win64-1
# wget -q https://github.com/protocolbuffers/protobuf/releases/download/v3.12.2/protoc-3.12.2-osx-x86_64.zip 
# unzip protoc-3.12.2-osx-x86_64.zip 
# cd ../

git clone --depth 1 https://github.com/tensorflow/models
cd models/research

## CHANGE THE DIRECTORY OF THE PROTOC THAT YOU DOWNLOAD EARLIER ##
C:/Users/AI/Desktop/face-algorithm/protoc-3.12.4-win64/bin/protoc.exe object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .

cd ../../
## EDIT THIS BASED ON COMPATABILITY OF THE TENSORFLOW DEPENDENCIES ##
python tensorflow-version.py
pip install models/research
pip install tensorflow==2.9.0
pip install tensorflow-addons==0.19.0
