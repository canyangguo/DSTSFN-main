## Dynamic Spatiotemporal Straight-Flow Network

## make dir
```
mkdir params
mkdir log
```

## run
```
python main.py --config config/PEMS08.json --gpu 0 --num_of_layers 4 --num_of_latents 32 --model_name DSTSFN
```

## requirements
* python 3.9.12
* torch 1.11.0
* numpy 1.20.1
* pandas 1.4.2