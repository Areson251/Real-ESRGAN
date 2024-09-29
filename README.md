| ![Image 1](assets/img_003_SRF_4_LR.png) | ![Image 2](assets/img_003_SRF_4_HR.png) | ![Image 3](assets/img_003_SRF_4_LR_out.png) |
|------------------------|------------------------|------------------------|
| ![Image 1](assets/img_012_SRF_4_LR.png) | ![Image 2](assets/img_012_SRF_4_HR.png) | ![Image 3](assets/img_012_SRF_4_LR_out.png) |



### Download dataset
```
wget "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
wget "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"

unzip DIV2K_train_HR.zip -d images/Div2K
unzip DIV2K_valid_HR.zip -d images/Div2K
```

### Prepare for training
**Use Docker for setup**
```
cd docker
chmod +x build.sh start.sh into.sh

./build.sh
./start.sh
./into.sh
```

**[Optional] Crop to sub-images**
For faster IO and processing

```
 python scripts/extract_subimages.py --input images/Div2K/train --output images/Div2K/train_sub --crop_size 400 --step 200

 python scripts/extract_subimages.py --input images/Div2K/valid --output images/Div2K/valid_sub --crop_size 400 --step 200
```

**Prepare txt for meta information** with image paths

```
 python scripts/generate_meta_info.py --input images/Div2K/train_sub --root images/Div2K --meta_info images/Div2K/meta_info/meta_info_Div2K_sub.txt
```

### Training

**Download pretrained weights**
```
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models

wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_netD.pth -P experiments/pretrained_models
```

**Modify [config](options/finetune_realesrgan_x4plus.yml)***

```
datasets:
  train:
    name: Div2K
    type: RealESRGANDataset
    dataroot_gt: images/Div2K/
    meta_info: images/Div2K/meta_info/meta_info_Div2K_sub.txt
    io_backend:
      type: disk
```

```
path:
  # use the pre-trained Real-ESRNet model
  pretrain_network_g: experiments/pretrained_models/RealESRGAN_x4plus.pth
  param_key_g: params_ema
  strict_load_g: true
  pretrain_network_d: experiments/pretrained_models/RealESRGAN_x4plus_netD.pth
  param_key_d: params
  strict_load_d: true
  resume_state: ~
```

```
logger:
  print_freq: 100
  save_checkpoint_freq: !!float 5e3
  use_tb_logger: true
  wandb:
    project: Real-ESRGAN
    resume_id: ~
```

**Debug**
```
python realesrgan/train.py -opt options/finetune_realesrgan_x4plus.yml --debug
```

#### Train!
```
python realesrgan/train.py -opt options/finetune_realesrgan_x4plus.yml --auto_resume
```
Training schedules is [here](https://wandb.ai/areson251/Real-ESRGAN)

### Inference
```
python inference_realesrgan.py -n RealESRGAN_x4plus --model_path experiments/finetune_RealESRGANx4plus_400k/models/net_g_20000.pth -i images/Set14/image_SRF_4_LR --output results/Set14
```
### Metrics
```
python metrics.py -gt_images images/Set14/image_SRF_4_HR -dt_images results/Set14
```

| Dataset | PSNR   | SSIM   | LPIPS  |
|---------|--------|--------|--------|
| Set5    | 24.0597| 0.6773 | 0.1745 |
| Set14   | 23.1149| 0.6052 | 0.2525 |


