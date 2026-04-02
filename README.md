# TIDES: Traffic Intelligence with DeepSeek-Enhanced Spatial-Temporal Prediction

> **IEEE Journal on Selected Areas in Communications, Vol. 44, 2026**  
> Chuanting Zhang, Haixia Zhang, Jingping Qiao, Zongzhang Li, Mohamed-Slim Alouini

TIDES is a novel LLM-based framework for urban wireless traffic prediction. It captures spatial-temporal correlations via a two-phase process: (1) spatial-aware region clustering, and (2) LLM-driven spatial-temporal learning with prompt engineering and cross-domain attention alignment.

## Architecture Overview

```
Input (B, T, N)
  ↓  Instance Normalization
  ↓  Patch Embedding
  ↓  Spatial Attention  ←── Adjacency Matrix (k-NN graph)
  ↓  Cross-Domain Attention (Reprogramming)  ←── LLM Word Embeddings
  ↓  LLM Forward Pass (frozen, with traffic-aware prompts)
  ↓  Feature Concatenation + Linear Mapping
  ↓  Prediction (B, pred_len, N)
```

Key innovations:
- **Region-aware modeling**: clusters base stations by traffic similarity + spatial autocorrelation (Local Moran's I)
- **Prompt-based traffic representation**: encodes traffic statistics (peak-to-average ratio, burstiness, rush hour intensity, etc.) into structured natural-language prompts
- **Spatial alignment via DeepSeek**: cross-domain attention lets the LLM leverage information from spatially neighboring cells without retraining the LLM

## Requirements

```bash
pip install torch torchvision torchaudio   # PyTorch >= 2.0 recommended (enables Flash Attention)
pip install transformers                    # HuggingFace (GPT-2, BERT, T5, LLaMA, DeepSeek)
pip install deepspeed                       # optional, for multi-GPU / ZeRO-2 training
pip install pandas numpy scikit-learn matplotlib
```

> For LLaMA or DeepSeek, download the model weights locally and set `--llm_model LLAMA` / `--llm_model deepseek` with `--llm_path` pointing to the local directory.

## Data

Place your traffic CSV under `datasets/<data_path>/traffic.csv`. The included dataset is:

```
datasets/zte4g/traffic.csv   # Real-world 4G/5G base station traffic (ZTE)
```

Expected format: rows are time steps (datetime index), columns are base station IDs.

## Quick Start

### Single GPU

```bash
bash run_optimized.sh
```

Or run directly:

```bash
python main.py \
  --is_training 1 \
  --gpu_ids 0 \
  --root_path ./datasets \
  --data_path zte4g \
  --model DSTraffic \
  --llm_model GPT2 \
  --llm_dim 768 \
  --llm_layers 16 \
  --features M \
  --seq_len 96 \
  --label_len 6 \
  --pred_len 4 \
  --batch_size 4 \
  --learning_rate 0.001 \
  --train_epochs 10 \
  --patience 10 \
  --d_model 16 \
  --d_ff 32 \
  --n_heads 8
```

### Switching LLM Backend

Edit `LLM_MODEL` in `run_optimized.sh` or pass the corresponding flags:

| LLM | `--llm_model` | `--llm_dim` | `--llm_layers` |
|-----|--------------|------------|---------------|
| GPT-2 | `GPT2` | 768 | 16 |
| BERT | `BERT` | 768 | 6 |
| T5 | `T5` | 512 | 6 |
| LLaMA | `LLAMA` | 4096 | 16 |
| DeepSeek | `deepseek` | 4096 | 16 |

### Multi-GPU (DeepSpeed ZeRO-2)

```bash
deepspeed --num_gpus 4 main.py \
  --deepspeed ds_config_zero2.json \
  [... same args as above ...]
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--seq_len` | 24 | Historical window length |
| `--pred_len` | 6 | Forecast horizon |
| `--label_len` | 6 | Decoder overlap length |
| `--d_model` | 16 | Model hidden dimension |
| `--n_heads` | 8 | Attention heads |
| `--d_ff` | 32 | Feed-forward dimension |
| `--dropout` | 0.1 | Dropout rate |
| `--num_neighbors` | 5 | k-NN graph neighbors per base station |
| `--gso_type` | `sym_norm_lap` | Graph shift operator type |
| `--loss` | `mse` | Loss: `mse`, `mae`, `huber`, `combined` |
| `--lradj` | `onecycle` | LR scheduler: `onecycle`, `cosine`, `step` |
| `--use_amp` | False | Enable bfloat16 mixed precision |

## Output

After training, results are saved to `./results/<run_id>/`:

```
best_model.pth          # Best checkpoint (lowest validation loss)
metrics_log.csv         # Epoch-wise MSE / MAE / MAPE / RMSE / LR
predictions.npz         # Test set predictions and ground truth
training_curves.png     # Loss, MAE, and LR plots
```

The test report prints per-horizon and per-station errors.

## Project Structure

```
TIDES/
├── main.py                          # Entry point, argument parsing, training setup
├── train.py                         # TIDESTrainer: train / validate / test loops
├── run_optimized.sh                 # Single-GPU training script
├── ds_config_zero2.json             # DeepSpeed ZeRO-2 config
├── models/
│   ├── DSTraffic.py                 # Base model
│   └── DSTraffic_FlashAttention.py  # Main model with Flash Attention
├── layers/
│   ├── Embed.py                     # Patch, positional, temporal embeddings
│   ├── StandardNorm.py              # Reversible instance normalization
│   ├── AutoCorrelation.py           # Auto-correlation module
│   └── Autoformer_EncDec.py         # Encoder/decoder blocks
├── datasets/
│   ├── data_factory.py              # Dataset class and data provider
│   └── zte4g/traffic.csv            # Included dataset
└── utils/
    ├── connectivity.py              # Haversine distance, k-NN adjacency matrix
    ├── timefeatures.py              # Temporal feature extraction
    └── tools.py                     # Graph shift operators, early stopping, metrics
```

## Citation

```bibtex
@article{zhang2026tides,
  title   = {{TIDES}: Traffic Intelligence with {DeepSeek}-Enhanced Spatial-Temporal Prediction},
  author  = {Zhang, Chuanting and Zhang, Haixia and Qiao, Jingping and Li, Zongzhang and Alouini, Mohamed-Slim},
  journal = {IEEE Journal on Selected Areas in Communications},
  volume  = {44},
  year    = {2026},
  doi     = {10.1109/JSAC.2025.3643397}
}
```
