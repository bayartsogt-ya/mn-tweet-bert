pwd
source env/bin/activate

python run_classifier.py \
  --task_name tweetmn \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir ./data/tweetmn/train \
  --bert_model ./models/uncased_bert_base_pytorch \
  --max_seq_length 64 \
  --train_batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 10.0 \
  --output_dir ./output/tweetmn-epoch-10/ 
