nohup python runner.py \
  --cuda_visible_devices 0,1 \
  --dataset_name nq_10 \
  --input_path retrieval/dataset/nq-open-10_total_documents_gold_at_0_bert.pkl \
  --model_name /home/akmoex/.cache/modelscope/hub/models/shakechen/Llama-2-7b-hf \
  --load_in_8bit \
  --use_training \
  --save_model \
  --output_dir output/rrag/Rrag-Llama-2-7b \
  --freeze_llm \
  --num_k 10 \
  --use_rrag \
  --use_evaluation \
  --save_results \
  > output/rrag/train.log 2>&1 &