#!/bin/bash

fps=$1 #define fps $1 means its the first parameter

if [[ -n "$fps" ]]; then
  for f in sample_videos/*; do

    filename=$(basename -- "$f")
    no_ext="${filename%.*}"

    echo "Processing $no_ext"

    ffmpeg -i $f -r $fps $"sample_images/$no_ext%03d.png"
  done
  echo "Done"
  
else
  echo "Please, add desired fps. Example: !bash video_to_images.sh 24"
fi

