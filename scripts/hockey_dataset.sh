
#!/bin/bash

#Hockey Dataset
echo "===> Preparing Hockey dataset..."
gdown --id 1200VOpfMFUys_IrWWumI6qUP3-4-1U1G
filename="HockeyFightsDATASET.zip"
src="/content/${filename}"
dst="/content/DATASETS"
mv $src ${dst}
f_name="${dst}/${filename}"
unzip -q $f_name -d $dst
rm $f_name

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
echo "===> Downloading Hockey ActionTubes..."
filename="HockeyFightsDATASET.zip"
gdown --id 1CiIzh99DeL_8JdkcUqIGUMjysCLJWNM8
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

#Pretrained models: TF

id="1oH7WtUMg-juZe3zdgiqcYGd26XmkjJ71"
echo "===> Downloading Initial model for Transfer learning: ${id}"
gdown --id $id
src="/content/$(ls -t | head -1)"
dst="/content/DATASETS/Pretrained_Models"
echo "===> Moving ${src} to ${dst}"
mv $src ${dst}