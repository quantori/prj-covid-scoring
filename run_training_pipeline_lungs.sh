python train.py \
    --dataset_dir dataset/lungs_segmentation\
    --model_name Unet \
    --input_size 384 384 \
    --encoder_name se_resnext101_32x4d \
    --loss_seg BCE \
    --optimizer Adam \
    --lr 0.0001 \
    --batch_size 24

python train.py \
    --dataset_dir dataset/lungs_segmentation\
    --model_name Unet++ \
    --input_size 384 384 \
    --encoder_name efficientnet-b1 \
    --loss_seg Jaccard \
    --optimizer Adam_amsgrad \
    --lr 0.001 \
    --batch_size 32

python train.py \
    --dataset_dir dataset/lungs_segmentation\
    --model_name DeepLabV3 \
    --input_size 512 512 \
    --encoder_name efficientnet-b0 \
    --loss_seg Dice \
    --optimizer AdamW_amsgrad \
    --lr 0.0005 \
    --batch_size 16

python train.py \
    --dataset_dir dataset/lungs_segmentation\
    --model_name DeepLabV3+ \
    --input_size 512 512 \
    --encoder_name efficientnet-b1 \
    --loss_seg BCE \
    --optimizer AdamW \
    --lr 0.0005 \
    --batch_size 20

python train.py \
    --dataset_dir dataset/lungs_segmentation\
    --model_name FPN \
    --input_size 544 544 \
    --encoder_name efficientnet-b0 \
    --loss_seg BCE \
    --optimizer Adam_amsgrad \
    --lr 0.001 \
    --batch_size 32

python train.py \
    --dataset_dir dataset/lungs_segmentation\
    --model_name Linknet \
    --input_size 480 480 \
    --encoder_name timm-regnetx_064 \
    --loss_seg BCE \
    --optimizer AdamW \
    --lr 0.0001 \
    --batch_size 24

python train.py \
    --dataset_dir dataset/lungs_segmentation\
    --model_name PSPNet \
    --input_size 480 480 \
    --encoder_name timm-regnety_064 \
    --loss_seg Dice \
    --optimizer Adam \
    --lr 0.0001 \
    --batch_size 40

python train.py \
    --dataset_dir dataset/lungs_segmentation\
    --model_name PAN \
    --input_size 512 512 \
    --encoder_name efficientnet-b0 \
    --loss_seg Jaccard \
    --optimizer Adam_amsgrad \
    --lr 0.001 \
    --batch_size 32

python train.py \
    --dataset_dir dataset/lungs_segmentation\
    --model_name MAnet \
    --input_size 512 512 \
    --encoder_name efficientnet-b2 \
    --loss_seg Dice \
    --optimizer Adam_amsgrad \
    --lr 0.001 \
    --batch_size 24