#!/bin/bash
#RLVSD Dataset
echo "===> Preparing RLVSD dataset..."
gdown --id 1tRZV2Iq1XC4HlATBNVdwc1Bg8EAjkZVn
filename="RLVSframes.zip"
src="/content/${filename}"
dst="/content/DATASETS"
mv $src ${dst}
f_name="${dst}/${filename}"
unzip -q $f_name -d $dst
rm $f_name

mkdir /content/DATASETS/RealLifeViolenceDataset
src="/content/DATASETS/frames"
dst="/content/DATASETS/RealLifeViolenceDataset"
mv $src ${dst}


##splits
echo "===> Downloading splits..."
gdown --id 106svvpxJDQV-eiFIbc3DUtqLrydXtQMm
filename="VioNetDB-splits.zip"
src="/content/${filename}"
dst="/content/DATASETS"
mv $src ${dst}
f_name="${dst}/${filename}"
unzip -q $f_name -d $dst
rm $f_name

#Action Tubes
echo "===> Downloading RLVSD ActionTubes..."
filename="RealLifeViolenceDataset.zip"
gdown --id 1o1ml94d7jPhK3lSD-mYJti5QvaBcUHjx
src="/content/${filename}"
dst="/content/DATASETS/ActionTubesV2"
mv $src ${dst}
f_name="${dst}/${filename}"
unzip -q $f_name -d $dst
rm $f_name

#Pretrained models

id="1oH7WtUMg-juZe3zdgiqcYGd26XmkjJ71"
echo "===> Downloading I3D model: ${id}"
gdown --id $id
src="/content/$(ls -t | head -1)"
dst="/content/DATASETS/Pretrained_Models"
echo "===> Moving ${src} to ${dst}"
mv $src ${dst}