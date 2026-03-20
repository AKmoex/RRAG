## 微调检索器
python finetune_retriever.py \
    --dataset_name 2wiki \
    --train_data_path 2wikimultihop/train.json \
    --test_data_path 2wikimultihop/dev.json \
    --model_name google-bert/bert-base-uncased \
    --train_batch_size 32 \
    --num_epoch 4 \
    --model_save_path /hdd-data/rrag/mine_1/2wiki/retrievers/2wiki_bert

## 提取特征
cd retrieval
HF_ENDPOINT=https://hf-mirror.com python feature_extraction.py \
    --dataset_name 2wiki \
    --train_data_path 2wikimultihop/train.json \
    --test_data_path 2wikimultihop/dev.json \
    --save_train_path /hdd-data/rrag/mine_1/2wiki/features/2wikimultihop_train_bert.pkl \
    --save_test_path /hdd-data/rrag/mine_1/2wiki/features/2wikimultihop_dev_bert.pkl \
    --model_name /hdd-data/rrag/mine_1/2wiki/retrievers/2wiki_bert