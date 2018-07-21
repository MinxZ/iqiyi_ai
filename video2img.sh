# -r 2 : 2 frames per 1 second

ffmpeg -i IQIYI_VID_TRAIN_0000001.mp4 2>&1 | grep "Duration"
ffmpeg -i IQIYI_VID_TRAIN/IQIYI_VID_TRAIN_0000001.mp4 2>&1 | grep "Duration"| cut -d ' ' -f 4 | sed s/,// | sed 's@\..*@@g' | awk '{ split($1, A, ":"); split(A[3], B, "."); print 3600*A[1] + 60*A[2] + B[1] }'

cd IQIYI_VID_TRAIN
mkdir ../png_train
for x in IQIYI_VID_TRAIN_*.mp4;
  do echo $x_1;
  ffmpeg -i $x -vf fps=1/2 ../png_train/$x'_'%04d.png
done

cd ../IQIYI_VID_VAL
mkdir ../png_val
for x in IQIYI_VID_VAL_*.mp4;
  do echo $x_1;
  ffmpeg -i $x -vf fps=1/2 ../png_val/$x'_'%04d.png
done
