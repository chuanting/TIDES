import os
import csv
import logging
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import math, types


class TIDESTrainer:
    """Training and evaluation system for wireless traffic prediction"""

    def __init__(self, args, device, model, train_loader, vali_loader, test_loader,
                 optimizer, scheduler, path):
        self.args = args
        self.device = device
        self.model = model
        self.train_loader = train_loader
        self.vali_loader = vali_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_dir = path

        self.results_dir = os.path.join(path, "results")
        self.logs_dir = os.path.join(path, "logs")

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Loss functions
        self.mse_criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()
        self.huber_criterion = nn.SmoothL1Loss(beta=0.5)

        # Metrics logging (CSV + file logger)
        self._metrics_csv_path = os.path.join(self.logs_dir, 'metrics.csv')
        self._metrics_fieldnames = ['epoch', 'phase', 'loss', 'mse', 'mae', 'mape', 'rmse', 'lr']
        with open(self._metrics_csv_path, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=self._metrics_fieldnames).writeheader()

        self._file_logger = logging.getLogger('metrics')
        self._file_logger.setLevel(logging.INFO)
        if not self._file_logger.handlers:
            fh = logging.FileHandler(os.path.join(self.logs_dir, 'metrics.log'))
            fh.setFormatter(logging.Formatter('%(asctime)s  %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
            self._file_logger.addHandler(fh)

        # Performance tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.no_improvement_count = 0

        # Training statistics
        self.train_stats = {
            'epochs': [], 'train_loss': [], 'val_loss': [],
            'val_mae': [], 'learning_rates': []
        }

        if args.clip_grad:
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.max_grad_norm = 1.0 * (total_params ** 0.5) / 100
            print(f"Gradient clipping max_norm: {self.max_grad_norm:.4f}")
        else:
            self.max_grad_norm = None

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self):
        print(f"Starting training for {self.args.train_epochs} epochs")

        for epoch in range(1, self.args.train_epochs + 1):
            train_loss, train_metrics = self.train_epoch(epoch)
            val_loss, val_metrics, early_stop = self.validate(epoch)

            self.train_stats['epochs'].append(epoch)
            self.train_stats['train_loss'].append(train_loss)
            self.train_stats['val_loss'].append(val_loss)
            self.train_stats['val_mae'].append(val_metrics['mae'])
            self.train_stats['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

            print(
                f"Epoch {epoch}/{self.args.train_epochs} - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                f"Val MAE: {val_metrics['mae']:.6f}, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )

            if early_stop:
                print(f"Early stopping triggered after {epoch} epochs")
                break

        print("\nTraining completed. Running final evaluation...")
        test_results = self.test()
        self._save_training_curves()
        return test_results

    # ------------------------------------------------------------------
    # Train one epoch
    # ------------------------------------------------------------------

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        train_metrics = {'mse': 0, 'mae': 0, 'huber': 0}
        processed_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_x, batch_y, batch_x_mark, batch_y_mark in progress_bar:
            self.optimizer.zero_grad()

            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            adj = None
            if hasattr(self.args, 'adj_k'):
                adj = torch.tensor(self.args.adj_k, device=self.device)

            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat(
                [batch_y[:, :self.args.label_len, :], dec_inp], dim=1
            ).float().to(self.device)

            with torch.autocast(device_type=self.device.type, enabled=self.args.use_amp):
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, adj)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:].contiguous()
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].contiguous()

                mse_loss = self.mse_criterion(outputs, batch_y)
                huber_loss = self.huber_criterion(outputs, batch_y)
                mae_loss = self.mae_criterion(outputs, batch_y)

                if self.args.loss.lower() == 'mae':
                    loss = mae_loss
                elif self.args.loss.lower() == 'combined':
                    loss = 0.7 * mse_loss + 0.3 * huber_loss
                else:
                    loss = mse_loss

                train_metrics['mse'] += mse_loss.item()
                train_metrics['mae'] += mae_loss.item()
                train_metrics['huber'] += huber_loss.item()

            loss.backward()

            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            processed_batches += 1

            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })

        avg_loss = total_loss / processed_batches
        for k in train_metrics:
            train_metrics[k] /= processed_batches

        self._log_metrics(epoch, 'train', loss=avg_loss, mse=train_metrics['mse'],
                          mae=train_metrics['mae'], lr=self.optimizer.param_groups[0]['lr'])

        return avg_loss, train_metrics

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_trues = []

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in self.vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                adj = None
                if hasattr(self.args, 'adj_k'):
                    adj = torch.tensor(self.args.adj_k, device=self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1
                ).float().to(self.device)

                with torch.autocast(device_type=self.device.type, enabled=self.args.use_amp):
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, adj)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs.float()[:, -self.args.pred_len:, f_dim:].contiguous()
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].contiguous()

                mse_loss = self.mse_criterion(outputs, batch_y)
                huber_loss = self.huber_criterion(outputs, batch_y)
                mae_loss = self.mae_criterion(outputs, batch_y)

                if self.args.loss.lower() == 'mae':
                    loss = mae_loss
                elif self.args.loss.lower() == 'combined':
                    loss = 0.7 * mse_loss + 0.3 * huber_loss
                else:
                    loss = mse_loss

                total_loss += loss.item()
                all_preds.append(outputs.cpu().numpy())
                all_trues.append(batch_y.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_trues = np.concatenate(all_trues, axis=0)

        mse = np.mean((all_preds - all_trues) ** 2)
        mae = np.mean(np.abs(all_preds - all_trues))
        epsilon = 1e-6
        mape = np.mean(np.abs((all_trues - all_preds) / (np.abs(all_trues) + epsilon)))
        rmse = np.sqrt(mse)

        avg_loss = total_loss / len(self.vali_loader)

        self._log_metrics(epoch, 'val', loss=avg_loss, mse=mse, mae=mae, mape=mape, rmse=rmse)

        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.best_epoch = epoch
            self.no_improvement_count = 0
            self._save_state(os.path.join(self.model_dir, "best_model"))
            print(f"New best model saved at epoch {epoch} with validation loss: {avg_loss:.4f}")
        else:
            self.no_improvement_count += 1

        val_metrics = {'mse': mse, 'mae': mae, 'mape': mape, 'rmse': rmse}
        return avg_loss, val_metrics, (self.no_improvement_count >= self.args.patience)

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test(self):
        self._load_state(os.path.join(self.model_dir, "best_model"))
        self.save_checkpoint(os.path.join(self.model_dir, "best_model"))

        self.model.eval()
        all_preds = []
        all_trues = []

        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in self.test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                adj = None
                if hasattr(self.args, 'adj_k'):
                    adj = torch.tensor(self.args.adj_k, device=self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len, :], dec_inp], dim=1
                ).float().to(self.device)

                with torch.autocast(device_type=self.device.type, enabled=self.args.use_amp):
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, adj)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs.float()[:, -self.args.pred_len:, f_dim:].contiguous()
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].contiguous()

                all_preds.append(outputs.cpu().numpy())
                all_trues.append(batch_y.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_trues = np.concatenate(all_trues, axis=0)

        mae = np.mean(np.abs(all_preds - all_trues))
        mse = np.mean((all_preds - all_trues) ** 2)
        rmse = np.sqrt(mse)
        epsilon = 1e-6
        mape = np.mean(np.abs((all_trues - all_preds) / (np.abs(all_trues) + epsilon)))
        smape = np.mean(
            2 * np.abs(all_preds - all_trues) / (np.abs(all_preds) + np.abs(all_trues) + epsilon)
        )

        horizon_errors = [
            np.mean(np.abs(all_preds[:, i, :] - all_trues[:, i, :]))
            for i in range(self.args.pred_len)
        ]
        station_errors = [
            np.mean(np.abs(all_preds[:, :, i] - all_trues[:, :, i]))
            for i in range(all_preds.shape[2])
        ]

        print("\n======= Test Results =======")
        print(f"MSE:   {mse:.4f}")
        print(f"MAE:   {mae:.4f}")
        print(f"RMSE:  {rmse:.4f}")
        print(f"MAPE:  {mape:.4f}")
        print(f"SMAPE: {smape:.4f}")
        print("\nError by prediction step:")
        for i, err in enumerate(horizon_errors):
            print(f"  Step {i + 1}: {err:.4f}")

        result_dict = {
            'mse': mse, 'mae': mae, 'rmse': rmse, 'mape': mape, 'smape': smape,
            'horizon_errors': horizon_errors, 'station_errors': station_errors,
            'predictions': all_preds, 'ground_truth': all_trues
        }

        output_file = os.path.join(self.results_dir, "test_results.npz")
        np.savez_compressed(output_file, **result_dict)
        print(f"Test results saved to {output_file}")

        return result_dict

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(self, path):
        """Save only model weights."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "model_weights.pt"))

    def _save_state(self, path):
        """Save model + optimizer + scheduler state for resuming."""
        os.makedirs(path, exist_ok=True)
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            state['scheduler'] = self.scheduler.state_dict()
        torch.save(state, os.path.join(path, 'training_state.pt'))

    def _load_state(self, path):
        """Restore model + optimizer + scheduler state."""
        state = torch.load(os.path.join(path, 'training_state.pt'), map_location=self.device)
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        if self.scheduler is not None and 'scheduler' in state:
            self.scheduler.load_state_dict(state['scheduler'])

    # ------------------------------------------------------------------
    # Logging / plotting helpers
    # ------------------------------------------------------------------

    def _log_metrics(self, epoch, phase, loss=None, mse=None, mae=None,
                     mape=None, rmse=None, lr=None):
        row = {
            'epoch': epoch, 'phase': phase,
            'loss': f'{loss:.6f}' if loss is not None else '',
            'mse':  f'{mse:.6f}'  if mse  is not None else '',
            'mae':  f'{mae:.6f}'  if mae  is not None else '',
            'mape': f'{mape:.6f}' if mape is not None else '',
            'rmse': f'{rmse:.6f}' if rmse is not None else '',
            'lr':   f'{lr:.8f}'   if lr   is not None else '',
        }
        with open(self._metrics_csv_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=self._metrics_fieldnames).writerow(row)
        self._file_logger.info(
            '[%s] epoch=%d  loss=%s  mse=%s  mae=%s  mape=%s  rmse=%s  lr=%s',
            phase, epoch,
            row['loss'], row['mse'], row['mae'], row['mape'], row['rmse'], row['lr']
        )

    def _save_training_curves(self):
        df = pd.DataFrame(self.train_stats)

        plt.figure(figsize=(10, 6))
        plt.plot(df['epochs'], df['train_loss'], label='Training Loss')
        plt.plot(df['epochs'], df['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.logs_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(df['epochs'], df['val_mae'], label='Validation MAE', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title('Validation MAE')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.logs_dir, 'mae_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(df['epochs'], df['learning_rates'], label='Learning Rate', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.savefig(os.path.join(self.logs_dir, 'lr_schedule.png'), dpi=300, bbox_inches='tight')
        plt.close()

        df.to_csv(os.path.join(self.logs_dir, 'training_stats.csv'), index=False)


class AdditionalTrainingFeatures:
    """
    Additional training features that can be added to enhance the model
    """

    @staticmethod
    def add_gradient_checkpointing(model):
        """Enable gradient checkpointing for memory efficiency"""
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            return True

        def make_checkpointed(module):
            original_forward = module.forward

            def checkpointed_forward(*args, **kwargs):
                return torch.utils.checkpoint.checkpoint(original_forward, *args, **kwargs)

            module.forward = checkpointed_forward

        for name, module in model.named_children():
            if any(x in name.lower() for x in ['encoder', 'decoder', 'transformer', 'attention']):
                if isinstance(module, nn.Module) and sum(p.numel() for p in module.parameters()) > 1000000:
                    make_checkpointed(module)
                    return True

        return False

    @staticmethod
    def add_parameter_efficient_finetuning(model, adapter_size=64, scale=1.0):
        try:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=adapter_size,
                lora_alpha=scale,
                target_modules=["query", "key", "value", "out_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, lora_config)
            return model, True

        except ImportError:
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and any(x in name for x in ['query', 'key', 'value']):
                    in_features, out_features = module.in_features, module.out_features

                    module.lora_A = nn.Parameter(torch.zeros(adapter_size, in_features))
                    module.lora_B = nn.Parameter(torch.zeros(out_features, adapter_size))

                    nn.init.kaiming_uniform_(module.lora_A, a=math.sqrt(5))
                    nn.init.zeros_(module.lora_B)

                    original_forward = module.forward

                    def forward_with_lora(self, x):
                        original_output = original_forward(x)
                        lora_output = (x @ self.lora_A.T) @ self.lora_B.T
                        return original_output + scale * lora_output

                    module.forward = types.MethodType(forward_with_lora, module)

            return model, False
