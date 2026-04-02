import numpy as np
import torch
import scipy.sparse as sp
from tqdm import tqdm


def calc_gso(dir_adj, gso_type):
    """
    Calculate a graph shift operator (GSO) based on the given adjacency matrix and GSO type.

    Args:
        dir_adj: Directed adjacency matrix
        gso_type: Type of graph shift operator (sym_norm_lap, rw_norm_lap, sym_renorm_adj, rw_renorm_adj)

    Returns:
        Graph shift operator as a sparse matrix
    """
    n_vertex = dir_adj.shape[0]

    # Ensure the adjacency matrix is in CSC format
    if not sp.issparse(dir_adj):
        dir_adj = sp.csc_matrix(dir_adj)
    elif dir_adj.format != 'csc':
        dir_adj = dir_adj.tocsc()

    id = sp.identity(n_vertex, format='csc')

    # Symmetrize the adjacency matrix
    adj = dir_adj + dir_adj.T.multiply(dir_adj.T > dir_adj) - dir_adj.multiply(dir_adj.T > dir_adj)

    # Add self-loops for renormalized variants
    if gso_type in ['sym_renorm_adj', 'rw_renorm_adj', 'sym_renorm_lap', 'rw_renorm_lap']:
        adj = adj + id

    # Compute the GSO based on the specified type
    if gso_type in ['sym_norm_adj', 'sym_renorm_adj', 'sym_norm_lap', 'sym_renorm_lap']:
        # Symmetric normalization
        row_sum = adj.sum(axis=1).A1
        row_sum_inv_sqrt = np.power(row_sum, -0.5)
        row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
        deg_inv_sqrt = sp.diags(row_sum_inv_sqrt, format='csc')
        sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)

        if gso_type in ['sym_norm_lap', 'sym_renorm_lap']:
            gso = id - sym_norm_adj  # Laplacian
        else:
            gso = sym_norm_adj

    elif gso_type in ['rw_norm_adj', 'rw_renorm_adj', 'rw_norm_lap', 'rw_renorm_lap']:
        # Random walk normalization
        row_sum = np.sum(adj, axis=1).A1
        row_sum_inv = np.power(row_sum, -1)
        row_sum_inv[np.isinf(row_sum_inv)] = 0.
        deg_inv = np.diag(row_sum_inv)
        rw_norm_adj = deg_inv.dot(adj)

        if gso_type in ['rw_norm_lap', 'rw_renorm_lap']:
            gso = id - rw_norm_adj  # Laplacian
        else:
            gso = rw_norm_adj
    else:
        raise ValueError(f'{gso_type} is not defined.')

    return gso


def adjust_learning_rate(accelerator, optimizer, scheduler, epoch, args, printout=True):
    """
    Adjust the learning rate based on the specified strategy

    Args:
        accelerator: Accelerator for distributed training
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        args: Training arguments
        printout: Whether to print the learning rate change
    """
    # Define learning rate adjustment strategies
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}

    # Apply the learning rate adjustment if applicable for this epoch
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            if accelerator is not None:
                accelerator.print(f'Updating learning rate to {lr}')
            else:
                print(f'Updating learning rate to {lr}')


class EarlyStopping:
    """
    Early stopping handler to terminate training when validation loss stops improving
    """

    def __init__(self, accelerator=None, patience=7, verbose=False, delta=0, save_mode=True):
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.save_mode = save_mode

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.accelerator is None:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            if self.accelerator is not None:
                self.accelerator.print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
            else:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')

        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(model)
            self.accelerator.save_state(path + '/' + 'checkpoint')
        else:
            torch.save(model.state_dict(), path + '/' + 'checkpoint')
        self.val_loss_min = val_loss


class StandardScaler:
    """
    Standard scaler for data normalization
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def find_top_k(arr, k):
    """
    Find the top-k largest values in each row of a 2D array

    Args:
        arr: Input 2D numpy array
        k: Number of largest values to find

    Returns:
        Tuple of (sorted_indices, sorted_values) - the indices and values of the top-k elements
    """
    # print('CHUANTING ZHANG', arr, k)
    # Use argpartition to efficiently find indices of top-k elements
    unsorted_indices = np.argpartition(arr, -k, axis=1)[:, -k:]

    # Get the corresponding values
    unsorted_values = arr[np.arange(arr.shape[0])[:, None], unsorted_indices]

    # Sort in descending order
    sorted_order = np.argsort(-unsorted_values, axis=1)

    # Rearrange indices and values according to sorted order
    sorted_indices = np.take_along_axis(unsorted_indices, sorted_order, axis=1)
    sorted_values = np.take_along_axis(unsorted_values, sorted_order, axis=1)

    return sorted_indices, sorted_values


def vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric, adj):
    """
    Validation function to evaluate model performance

    Args:
        args: Training arguments
        accelerator: Accelerator for distributed validation
        model: Model to evaluate
        vali_data: Validation dataset
        vali_loader: DataLoader for validation data
        criterion: Loss function
        mae_metric: MAE metric
        adj: Adjacency matrix

    Returns:
        total_loss: Average loss
        total_mae_loss: Average MAE loss
        final_pred: Predictions
        final_truth: Ground truth values
    """
    total_loss = []
    total_mae_loss = []
    model.eval()

    pred_list = []
    truth_list = []

    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(vali_loader)):
            # Prepare data
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # Prepare decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                accelerator.device)

            # Forward pass
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, adj)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, adj)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, adj)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, adj)

            # Gather distributed outputs
            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))

            # Extract relevant dimensions
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

            # Store predictions and ground truth
            pred = outputs.detach()
            true = batch_y.detach()

            pred_list.append(pred.cpu().numpy())
            truth_list.append(true.cpu().numpy())

            # Calculate losses
            loss = criterion(pred, true)
            mae_loss = mae_metric(pred, true)

            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())

    # Calculate average losses
    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)

    # Combine predictions and ground truth
    final_pred = np.concatenate(pred_list, axis=0)
    final_truth = np.concatenate(truth_list, axis=0)

    model.train()
    return total_loss, total_mae_loss, final_pred, final_truth