python train.py \
    --dataset_dir dataset/covid_segmentation_single_crop \
    --model_name Unet \
    --input_size 544 544 \
    --encoder_name dpn98 \
    --loss_seg Dice \
    --loss_cls SL1 \
    --optimizer Adam_amsgrad \
    --lr 0.0001 \
    --batch_size 12 \
    --use_cls_head

python train.py \
    --dataset_dir dataset/covid_segmentation_single_crop \
    --model_name Unet++ \
    --input_size 480 480 \
    --encoder_name timm-regnety_032 \
    --loss_seg Jaccard \
    --loss_cls SL1 \
    --optimizer Adam_amsgrad \
    --lr 0.0001 \
    --batch_size 20 \
    --use_cls_head

python train.py \
    --dataset_dir dataset/covid_segmentation_single_crop \
    --model_name DeepLabV3 \
    --input_size 480 480 \
    --encoder_name efficientnet-b2 \
    --loss_seg Lovasz \
    --loss_cls SL1 \
    --optimizer Adam \
    --lr 0.01 \
    --batch_size 10 \
    --use_cls_head

python train.py \
    --dataset_dir dataset/covid_segmentation_single_crop \
    --model_name DeepLabV3+ \
    --input_size 480 480 \
    --encoder_name timm-regnetx_032 \
    --loss_seg Jaccard \
    --loss_cls L1 \
    --optimizer Adam \
    --lr 0.001 \
    --batch_size 36 \
    --use_cls_head

python train.py \
    --dataset_dir dataset/covid_segmentation_single_crop \
    --model_name FPN \
    --input_size 384 384 \
    --encoder_name efficientnet-b0 \
    --loss_seg Dice \
    --loss_cls SL1 \
    --optimizer AdamW_amsgrad \
    --lr 0.0005 \
    --batch_size 64 \
    --use_cls_head

python train.py \
    --dataset_dir dataset/covid_segmentation_single_crop \
    --model_name Linknet \
    --input_size 416 416 \
    --encoder_name efficientnet-b0 \
    --loss_seg Dice \
    --loss_cls BCE \
    --optimizer AdamW \
    --lr 0.01 \
    --batch_size 48 \
    --use_cls_head

python train.py \
    --dataset_dir dataset/covid_segmentation_single_crop \
    --model_name PSPNet \
    --input_size 384 384 \
    --encoder_name efficientnet-b0 \
    --loss_seg Dice \
    --loss_cls BCE \
    --optimizer AdamW \
    --lr 0.001 \
    --batch_size 96 \
    --use_cls_head

python train.py \
    --dataset_dir dataset/covid_segmentation_single_crop \
    --model_name PAN \
    --input_size 416 416 \
    --encoder_name se_resnet50 \
    --loss_seg BCE \
    --loss_cls SL1 \
    --optimizer AdamW_amsgrad \
    --lr 0.005 \
    --batch_size 48 \
    --use_cls_head

python train.py \
    --dataset_dir dataset/covid_segmentation_single_crop \
    --model_name MAnet \
    --input_size 544 544 \
    --encoder_name timm-regnetx_064 \
    --loss_seg Dice \
    --loss_cls L1 \
    --optimizer RMSprop \
    --lr 0.0001 \
    --batch_size 18 \
    --use_cls_head