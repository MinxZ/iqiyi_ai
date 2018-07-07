# -r 2 : 2 frames per 1 second
for
ffmpeg -i IQIYI_VID_TRAIN_0000002.mp4 -r 2 ../output_%04d.png

for x in IQIYI_VID_TRAIN_000000*.mp4;
  do echo $x_1;
  ffmpeg -i $x -r 1 ../$x'_'%04d.png
done

for x in IQIYI_VID_TRAIN_000000*.mp4;
  do echo $x'_'%04d.png;
done
