# Training Step
## Trainig Parameters
- Training parameters are organized with [Hydra](https://hydra.cc/docs/configure_hydra/intro/).
- Configuration file is under ```conf/``` directory and it looks like this..
```yaml
net_type: 'disnet'
min_epoch: 1000
max_epoch: 2000
batch: 2
input_size: 1280
device: [0,1,2,3]
gtnet_path: 'saved_models/best_models/gtnet.ckpt'
disnet_path: 'saved_models/best_models/disnet.ckpt'
dataset:
    augmentation: True
    train:
        image_path: '../data/DIS5K/DIS-TR/im'
        mask_path: '../data/DIS5K/DIS-TR/gt'
        num_workers: 8
    val:
        image_path: '../data/DIS5K/DIS-VD/im'
        mask_path: '../data/DIS5K/DIS-VD/gt'
        num_workers: 4

```
## Code
```python
python train.py
```

# Inference
## Code
```python
python inference.py --img_name sample.jpg --device=cuda:1 --save_path inference_images
```
