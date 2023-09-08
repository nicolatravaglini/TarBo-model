#!/bin/bash

# creo per ogni video in video/ una cartella con il suo nome in dataset/training_set e in dataset/test_set
videos=$(ls ./drive/MyDrive/tarbo)
for video in $videos
do
  video_name=$(echo "$video" | cut -d "." -f 1)
  echo "$video_name"
  mkdir "dataset/training_set/$video_name"
  mkdir "dataset/test_set/$video_name"
done

# per ogni video estraggo ed inserisco i suoi frame nella apposita cartella di training
for video in $videos
do
  video_name=$(echo "$video" | cut -d "." -f 1)
  ffmpeg -i ./drive/MyDrive/tarbo/"$video" -vf scale=240:426 ./dataset/training_set/"$video_name"/%03d.png -hide_banner
done

# scelgo randomicamente 1/5 del training_set da spostare in test_set per ogni video
dirs_video=$(ls ./dataset/training_set)
for dir_video in $dirs_video
do
  videos=$(ls ./dataset/training_set/"$dir_video")
  for video in $videos
  do
    video_name=$(echo "$video" | cut -d "." -f 1)
    if [ $((1 + ($RANDOM % 5))) == 1 ]
    then
      mv ./dataset/training_set/"$dir_video"/"$video" ./dataset/test_set/"$dir_video"
    fi
  done
done