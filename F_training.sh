# mkdir training
# cd traning
# mkdir train
# cd ../../
## SET THE NUMBER OF STEPS ##
python models/research/object_detection/model_main_tf2.py --pipeline_config_path=models/mymodel/pipeline_file.config --model_dir=training/ --alsologtostderr --num_train_steps=20000 --sample_1_of_n_eval_examples=1