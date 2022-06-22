
#!/bin/bash

#RWF dataset
echo "===> Preparing RWF-2000 dataset..."
# cp  "/content/drive/My Drive/VIOLENCE DATA/RWF-2000.zip" "/content/DATASETS"
# unzip -q /content/DATASETS/RWF-2000.zip -d /content/DATASETS
# rm /content/DATASETS/RWF-2000.zip
id="1sJFv-A-mbUFCcNflgXeCYeDNgGQixUXC"
gdown --id $id
filename="RWF-2000.zip"
src="/content/${filename}"
dst="/content/DATASETS"
mv $src ${dst}
unzip -q /content/DATASETS/RWF-2000.zip -d /content/DATASETS
rm /content/DATASETS/RWF-2000.zip

#Action Tubes
# echo "===> Downloading RWF-2000 ActionTubes..."
# filename="RWF-2000.zip"
# src="/content/drive/My Drive/VIOLENCE DATA/Tubes/ActionTubesV2/${filename}"
# dst="/content/DATASETS/ActionTubesV2"
# f_name="${dst}/${filename}"
# cp -r "$src" $dst
# unzip -q $f_name -d $dst
# rm $f_name

echo "===> Downloading RWF-2000 ActionTubes..."
id="13aVSz9j7XV4mqhcNAn0I7JNNteUYEV2A"
gdown --id $id
filename="RWF-2000.zip"
src="/content/${filename}"
dst="/content/DATASETS/ActionTubesV2"
f_name="${dst}/${filename}"
cp -r "$src" $dst
unzip -q $f_name -d $dst
rm $f_name

echo "===> Downloading RWF-2000 ActionTubes Scored..."
id="10nC0l_JQ3sVzoJlHpJpkhZfhe_wHLwFO"
gdown --id $id
filename="RWF-2000.zip"
src="/content/${filename}"
mkdir '/content/DATASETS/ActionTubesV2Scored/RWF-2000'
dst="/content/DATASETS/ActionTubesV2Scored/RWF-2000"
f_name="${dst}/${filename}"
cp -r "$src" $dst
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

#Restore training
id="1-RDg150Os9C4ZxynJOUBgUA6mmH8PQWs"
echo "===> Downloading ${id}"
gdown --id $id
src="/content/$(ls -t | head -1)"
dst="/content/DATASETS/Pretrained_Models"
echo "===> Moving ${src} to ${dst}"
mv $src ${dst}
