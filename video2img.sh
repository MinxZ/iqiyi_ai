# -r 2 : 2 frames per 1 second

for x in IQIYI_VID_TRAIN_*.mp4;
  do echo $x_1;
  ffmpeg -i $x -r 1 ../$x'_'%04d.png
done
