#!/bin/sh
# 单卡 GPU 训练脚本

MODEL="DSTraffic"
LLM_MODEL="GPT2"
DATA_PATH="zte4g"
PRED_LEN=4
SEQ_LEN=96
BATCH_SIZE=4
EPOCHS=10
LR=0.001
GPU_ID=0
D_MODEL=16
D_FF=32
N_HEADS=8

# LLM 维度配置
if [ "$LLM_MODEL" = "GPT2" ]; then
  LLM_DIM=768;  LLM_LAYERS=16
elif [ "$LLM_MODEL" = "BERT" ]; then
  LLM_DIM=768;  LLM_LAYERS=6
elif [ "$LLM_MODEL" = "LLAMA" ]; then
  LLM_DIM=4096; LLM_LAYERS=16
elif [ "$LLM_MODEL" = "deepseek" ]; then
  LLM_DIM=4096; LLM_LAYERS=16
elif [ "$LLM_MODEL" = "T5" ]; then
  LLM_DIM=512;  LLM_LAYERS=6
else
  echo "Unknown LLM model: $LLM_MODEL"; exit 1
fi

echo "==============================================="
echo "Single-GPU Training"
echo "  Model: $MODEL | LLM: $LLM_MODEL ($LLM_DIM, $LLM_LAYERS layers)"
echo "  Data: $DATA_PATH | Seq: $SEQ_LEN -> Pred: $PRED_LEN"
echo "  Batch: $BATCH_SIZE | Epochs: $EPOCHS | LR: $LR | GPU: $GPU_ID"
echo "==============================================="

python main.py \
  --is_training 1 \
  --gpu_ids "$GPU_ID" \
  --root_path ./datasets \
  --data_path "$DATA_PATH" \
  --model "$MODEL" \
  --llm_model "$LLM_MODEL" \
  --llm_dim "$LLM_DIM" \
  --llm_layers "$LLM_LAYERS" \
  --features M \
  --seq_len "$SEQ_LEN" \
  --label_len 6 \
  --pred_len "$PRED_LEN" \
  --batch_size "$BATCH_SIZE" \
  --learning_rate "$LR" \
  --train_epochs "$EPOCHS" \
  --patience 10 \
  --d_model "$D_MODEL" \
  --d_ff "$D_FF" \
  --n_heads "$N_HEADS"

echo "Training complete!"
