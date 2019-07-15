#!/bin/bash

kps="_keypoints"
for i in {500..600}
do
  path="data/sample_images/"
  cxk="popping_dance"
  filename="$cxk$i"

  echo "Processing $filename"
  #echo json/$filename$kps.json
  python demo.py --img_path "$path$filename".png -- json_path data/json/$filename$kps.json
done
#for f in sample_images/*; do

	#filename=$(basename -- "$f")
  #no_ext="${filename%.*}$kps"
  
  #echo "Processing $no_ext"
  
  #python demo.py --img_path $f \
                     #--json_path json/$no_ext.json  
  
#done

echo "Done"