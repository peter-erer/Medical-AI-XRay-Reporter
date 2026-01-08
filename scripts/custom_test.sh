python custom_test.py \
    --image_dir "data/iu_xray_val/images" \
    --new_ann_path "data/iu_xray_val/annotation.json" \
    --train_image_dir "data/iu_xray/images/" \
    --old_ann_path "data/iu_xray/annotation.json" \
    --load "results/iu_xray_new_4/model_best.pth" \
    --d_vf 2048 \
    --d_model 512 \
    --sample_method sample \
    --temperature 1 \
    --d_ff 512
    