#!/bin/bash

# Input folder where videos are located
input_folder="lab/"

# Output folder to save processed videos
output_folder="/lab/formatted/"

# Desired video resolution
width="480"
height="270"

# Desired frame rate
frame_rate="10"

# Loop through all video files in the input folder
for file in "$input_folder"/*
do
  # Get the filename without extension
  filename=$(basename -- "$file")
  filename="${filename%.*}"

  # Set the output filename
  output_file="$output_folder/${filename}_processed.mp4"

  # Use ffmpeg to change video resolution and frame rate
  ffmpeg -i "$file" -vf scale="$width":"$height" -r "$frame_rate" "$output_file"
done
