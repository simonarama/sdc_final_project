# to be compatible to the tensorflow 1.3.0, run this script in tensorflow 1.3.0 to export
# inference graph that can be used with tensorflow 1.3.0

python export_inference_graph.py \
    --input_type image_tensor \
    --trained_checkpoint_prefix ./train/model.ckpt-20522 \
    --output_directory ./result \
    --pipeline_config_path ssd_mobilenet_v2_coco_3_classes_annotated_real.config
    #--pipeline_config_path ssd_mobilenet_v2_coco_3_classes_annotated_sim.config
    #--pipeline_config_path faster_rcnn_resnet50_coco_4_classes.config
    #--pipeline_config_path ssd_mobilenet_v2_coco_4_classes.config