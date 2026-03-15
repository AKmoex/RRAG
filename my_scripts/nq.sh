## 微调检索器
cd retrieval
python finetune_retriever.py \
    --dataset_name nq_10 \
    --input_path 10_total_documents/nq-open-10_total_documents_gold_at_0.jsonl.gz \
    --model_name google-bert/bert-base-uncased \
    --train_batch_size 32 \
    --num_epoch 4 \
    --model_save_path ../output/rrag/retrievers/nq10_bert


## 提取特征
cd retrieval
HF_ENDPOINT=https://hf-mirror.com python feature_extraction.py \
    --dataset_name nq_10 \
    --input_path 10_total_documents/nq-open-10_total_documents_gold_at_0.jsonl.gz \
    --save_path ../output/rrag/features/nq-open-10_total_documents_gold_at_0_bert.pkl \
    --model_name google-bert/bert-base-uncased \
    --model_name ../output/rrag/retrievers/nq10_bert

## 训练和评估
nohup python runner.py \
  --cuda_visible_devices 0,1 \
  --dataset_name nq_10 \
  --input_path output/rrag/features/nq-open-10_total_documents_gold_at_0_bert.pkl \
  --model_name /home/akmoex/.cache/modelscope/hub/models/shakechen/Llama-2-7b-hf \
  --load_in_8bit \
  --use_training \
  --save_model \
  --output_dir output/rrag/llama-2-7b \
  --freeze_llm \
  --num_k 10 \
  --use_rrag \
  --use_evaluation \
  --save_results \
  > output/rrag/llama-2-7b/train.log 2>&1 &

