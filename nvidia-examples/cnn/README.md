
# ResNet50 scripts
These scripts implement the ResNet50 v1.5 CNN model and demonstrate efficient
single-node training on multi-GPU systems. They can be used for benchmarking, or
as a starting point for implementing and training your own network.

Common utilities for defining CNN networks and performing basic training are
located in the nvutils directory. The utilities are written in Tensorflow 2.0.
Use of nvutils is demonstrated in the model script (i.e. resnet.py). The scripts
support both Keras Fit/Compile and Custom Training Loop (CTL) modes with
Horovod.

## Training in Keras Fit/Compile mode
For the full training on 8 GPUs: 
```
mpiexec --allow-run-as-root --bind-to socket -np 8 \
  python resnet.py --num_iter=90 --iter_unit=epoch \
  --data_dir=/data/imagenet/train-val-tfrecord-480/ \
  --precision=fp16 --display_every=100 \
  --export_dir=/tmp
```

For the benchmark training on 8 GPUs: 
```
mpiexec --allow-run-as-root --bind-to socket -np 8 \
  python resnet.py --num_iter=400 --iter_unit=batch \
  --data_dir=/data/imagenet/train-val-tfrecord-480/ \
  --precision=fp16 --display_every=100 
```

## Predicting in Keras Fit/Compile mode
For predicting with previously saved mode in `/tmp`:
```
python resnet.py --predict --export_dir=/tmp
```

## Training in CTL (Custom Training Loop) mode
For the full training on 8 GPUs: 
```
mpiexec --allow-run-as-root --bind-to socket -np 8 \
  python resnet_ctl.py --num_iter=90 --iter_unit=epoch \
  --data_dir=/data/imagenet/train-val-tfrecord-480/ \
  --precision=fp16 --display_every=100 \
  --export_dir=/tmp
```

For the benchmark training on 8 GPUs: 
```
mpiexec --allow-run-as-root --bind-to socket -np 8 \
  python resnet_ctl.py --num_iter=400 --iter_unit=batch \
  --data_dir=/data/imagenet/train-val-tfrecord-480/ \
  --precision=fp16 --display_every=100 
```

## Predicting in CTL (Custom Training Loop) mode
For predicting with previously saved mode in `/tmp`:
```
python resnet_ctl.py --predict --export_dir=/tmp
```

## Other useful options
To use tensorboard (Note, `/tmp/some_dir` needs to be created by users):
```
--tensorboard_dir=/tmp/some_dir
```

To export saved model at the end of training (Note, `/tmp/some_dir` needs to be created by users):
```
--export_dir=/tmp/some_dir
```

To store checkpoints at the end of every epoch (Note, `/tmp/some_dir` needs to be created by users):
```
--log_dir=/tmp/some_dir
```

To enable XLA
```
--use_xla
```

To use DALI pipeline for data loading and preprocessing
```
--dali_mode=GPU #or
--dali_mode=CPU
```

