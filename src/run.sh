export MAX_LEN=512
export EPOCHS=20
export TRAIN_BATCH_SIZE=4
export VALIDATION_BATCH_SIZE=4
export LR=3e-5
export BERT_MODEL='bert'
export BERT_PATH='bert-base-uncased'
export TOKENIZER_PATH='bert-base-uncased'

python train.py