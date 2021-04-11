#!/bin/bash

mkdir -p data/
cd data/

# download VizWiz
mkdir -p vizwiz
cd vizwiz
wget https://ivc.ischool.utexas.edu/VizWiz_final/images/train.zip
wget https://ivc.ischool.utexas.edu/VizWiz_final/images/val.zip
wget https://ivc.ischool.utexas.edu/VizWiz_final/images/test.zip
wget https://ivc.ischool.utexas.edu/VizWiz_final/vqa_data/Annotations.zip
unzip train.zip
unzip val.zip
unzip test.zip
unzip Annotations.zip
rm train.zip
rm val.zip
rm test.zip
rm Annotations.zip
cd ..

# TODO: download QRPE dataset
# TODO: download VQA 2.0 (if needed)
# TODO: download VizWiz-Priv (if needed)
# TODO: download VISPR (if needed)
