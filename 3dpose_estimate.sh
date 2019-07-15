#!/bin/bash

video=$1 #define video name

kps="_keypoints"
for i in {1..110}
do
  path="data/sample_images/"
  var=$(printf "%03d" "$i")
  filename="$video$var"

  echo "Processing $filename"
  #echo json/$filename$kps.json
  python demo.py --img_path "$path$filename".png --json_path data/json/$filename$kps.json
done

echo "Done"