import argparse
import torch
import os
import numpy as np
import time
import random
import logging
from train import TIDESTrainer
from torch import optim
from torch.optim import lr_scheduler
from utils.connectivity import analyze_base_station_connectivity

from models import DSTraffic_FlashAttention
from datasets.data_factory import data_provider
from utils.tools import calc_gso, find_top_k

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCHDYNAMO_CAPTURE_SCALAR_OUTPUTS"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def add_arguments():
    parser = argparse.ArgumentParser(description='Wireless Traffic Forecasting')

    # Environment settings
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--gpu_ids', type=str, default='0', help='GPU id to use (single GPU)')
    parser.add_argument('--use_amp', action='store_false', help='disable automatic mixed precision')
    parser.add_argument('--clip_grad', action='store_true', help='clip gradients', default=True)
    parser.add_argument('--enhanced_prompt', action='store_true', help='use enhanced prompt', default=True)

    # Basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1, help='1=train, 0=eval only')
    parser.add_argument('--model_id', type=str, default='traffic_forecast')
    parser.add_argument('--model_comment', type=str, default='optimized')
    parser.add_argument('--model', type=str, default='DSTraffic')

    # Data loader
    parser.add_argument('--root_path', type=str, default='datasets/')
    parser.add_argument('--data_path', type=str, default='zte4g/')
    parser.add_argument('--features', type=str, default='M', help='[M, S, MS]')
    parser.add_argument('--target', type=str, default='370102001')
    parser.add_argument('--freq', type=str, default='15min')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--percent', type=int, default=100)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')

    # Forecasting task
    parser.add_argument('--seq_len', type=int, default=24)
    parser.add_argument('--label_len', type=int, default=6)
    parser.add_argument('--pred_len', type=int, default=6)

    # Model
    parser.add_argument('--enc_in', type=int, default=1)
    parser.add_argument('--dec_in', type=int, default=1)
    parser.add_argument('--c_out', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=16)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=32)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', action='store_true')
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)

    # LLM params
    parser.add_argument('--llm_model', type=str, default='CHRONOS')
    parser.add_argument('--llm_dim', type=int, default=768)
    parser.add_argument('--llm_layers', type=int, default=6)
    parser.add_argument('--trained_epochs', type=int, default=-1)

    # Optimization
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--train_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--lradj', type=str, default='circle')
    parser.add_argument('--pct_start', type=float, default=0.2)
    parser.add_argument('--cluster', type=int, default=0)
    parser.add_argument('--loss', type=str, default='mse', help='[mse, mae, combined]')
    parser.add_argument('--deepspeed_config', type=str, default='ds_config_zero2.json', help='DeepSpeed config path')

    # Graph settings
    parser.add_argument('--neighbor', type=int, default=1)
    parser.add_argument('--gso_type', type=str, default='sym_norm_lap',
                        choices=['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj'])

    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def get_device(args):
    """Select compute device: CUDA > MPS > CPU"""
    gpu_id = int(args.gpu_ids.split(',')[0])

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    logger.info(f"Using device: {device}")
    return device


def prepare_adjacency_matrix(args, device):
    logger.info("Calculating adjacency matrix")
    try:
        file_path = os.path.join(args.root_path, args.data_path, 'bs.csv')
        _, _, _, adj, ids = analyze_base_station_connectivity(file_path)
        adj = calc_gso(adj, args.gso_type)
        adj = torch.from_numpy(adj.toarray().astype(np.float32)).to(device)
        return adj, ids
    except Exception as e:
        logger.error(f"Error loading adjacency matrix: {e}")
        raise


def load_data(args):
    logger.info(f"Loading data from {args.root_path}/{args.data_path}")
    try:
        train_data, train_loader = data_provider(args, 'train')
        vali_data, vali_loader = data_provider(args, 'val')
        test_data, test_loader = data_provider(args, 'test')
        logger.info(f"Data loaded. Vertices: {args.n_vertex}")
        return train_data, train_loader, vali_data, vali_loader, test_data, test_loader
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def create_model(args, device):
    logger.info(f"Creating model: {args.model}")
    try:
        model = DSTraffic_FlashAttention.Model(args).float().to(device)
        # Disable the model's manual bfloat16 casting — it converts tensors but leaves
        # layer weights as float32, causing dtype mismatches. The LLM sub-blocks already
        # have their own torch.cuda.amp.autocast(dtype=bfloat16) guards internally.
        model.use_mixed_precision = False
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model: {total_params:,} total params, {trainable_params:,} trainable "
                    f"({trainable_params / total_params * 100:.2f}%)")
        return model
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        raise


def main():
    args = add_arguments()
    set_random_seed(args.seed)

    os.environ['CURL_CA_BUNDLE'] = ''
    if args.use_amp:
        torch.set_float32_matmul_precision('high')

    os.makedirs(args.checkpoints, exist_ok=True)

    device = get_device(args)

    logger.info("=" * 80)
    logger.info(f"Wireless Traffic Prediction - Model: {args.model}")
    logger.info(f"Data: {args.data_path}, Pred: {args.pred_len}, Seq: {args.seq_len}")
    logger.info(f"LLM: {args.llm_model} (Layers: {args.llm_layers})")
    logger.info("=" * 80)

    setting = '{}_{}_sl{}_ll{}_pl{}_dm{}_nh{}_df{}_{}'.format(
        args.model, args.llm_model,
        args.seq_len, args.label_len, args.pred_len,
        args.d_model, args.n_heads, args.d_ff, args.cluster
    )
    path = os.path.join(args.checkpoints, setting)
    os.makedirs(path, exist_ok=True)

    adj, args.station_ids = prepare_adjacency_matrix(args, device)
    args.n_vertex = len(args.station_ids)
    args.adj_k = adj  # store adj on device for trainer access

    train_data, train_loader, vali_data, vali_loader, test_data, test_loader = load_data(args)

    model = create_model(args, device)

    trained_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trained_parameters, lr=args.learning_rate)
    train_steps = len(train_loader)

    if args.lradj == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-8)
    elif args.lradj == 'circle':
        scheduler = lr_scheduler.OneCycleLR(
            optimizer=optimizer, steps_per_epoch=train_steps,
            pct_start=args.pct_start, epochs=args.train_epochs,
            max_lr=args.learning_rate
        )
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.train_epochs // 3, gamma=0.5)

    trainer = TIDESTrainer(
        args, device, model,
        train_loader, vali_loader, test_loader,
        optimizer, scheduler, path
    )

    start_time = time.time()

    if args.is_training:
        test_results = trainer.train()
    else:
        test_results = trainer.test()

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"Completed in {elapsed / 60:.2f} minutes")
    logger.info(f"Test MAE: {test_results['mae']:.6f}")
    logger.info(f"Test RMSE: {test_results['rmse']:.6f}")
    logger.info(f"Test MAPE: {test_results['mape']:.4f}")
    logger.info("=" * 80)
    logger.info("Done!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Error occurred: {e}")
        raise
