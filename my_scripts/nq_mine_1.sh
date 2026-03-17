## 微调检索器, 其实不需要动, 用之前调好的就行
cd retrieval
python finetune_retriever.py \
    --dataset_name nq_10 \
    --input_path 10_total_documents/nq-open-10_total_documents_gold_at_0.jsonl.gz \
    --model_name google-bert/bert-base-uncased \
    --train_batch_size 32 \
    --num_epoch 4 \
    --model_save_path /hdd-data/rrag/mine_1/retrievers/nq10_bert



## 提取特征
cd retrieval
HF_ENDPOINT=https://hf-mirror.com python feature_extraction.py \
    --dataset_name nq_10 \
    --input_path 10_total_documents/nq-open-10_total_documents_gold_at_0.jsonl.gz \
    --save_path /hdd-data/rrag/mine_1/features/nq-open-10_total_documents_gold_at_0_bert.pkl \
    --model_name google-bert/bert-base-uncased \
    --model_name /hdd-data/rrag/mine_1/retrievers/nq10_bert \
    --use_all_doc_avg_feature

## 训练和评估
nohup python runner.py \
  --cuda_visible_devices 0,1 \
  --dataset_name nq_10 \
  --input_path  /hdd-data/rrag/mine_1/features/nq-open-10_total_documents_gold_at_0_bert.pkl \
  --model_name /home/akmoex/.cache/modelscope/hub/models/shakechen/Llama-2-7b-hf \
  --load_in_8bit \
  --use_training \
  --save_model \
  --output_dir  /hdd-data/rrag/mine_1/llama-2-7b \
  --freeze_llm \
  --num_k 10 \
  --use_rrag \
  --use_evaluation \
  --save_results \
  --use_all_doc_avg_feature \
  >  /hdd-data/rrag/mine_1/llama-2-7b/train.log 2>&1 &

