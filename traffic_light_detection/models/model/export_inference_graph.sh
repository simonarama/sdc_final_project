python export_inference_graph.py \
    --input_type image_tensor \
    --trained_checkpoint_prefix ./train/model.ckpt-21746 \
    --output_directory ./result \
    --pipeline_config_path ssd_mobilenet_v2_coco_3_classes_annotated_sim.config
    #--pipeline_config_path faster_rcnn_resnet50_coco_4_classes.config
    #--pipeline_config_path ssd_mobilenet_v2_coco_4_classes.config