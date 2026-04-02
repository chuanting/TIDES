import torch
import torch.nn as nn
import numpy as np


class Chronos2RobustScaling(nn.Module):
    """
    Chronos-2 Robust Scaling implementation

    按照论文3.1节的实现:
    1. 标准化: z = (x - μ) / σ
    2. Log-like变换: y = arcsinh(z)

    这种鲁棒缩放能更好地处理极端值和异常值
    """

    def __init__(self, eps=1e-6):
        """
        Args:
            eps: 数值稳定性的小常数
        """
        super(Chronos2RobustScaling, self).__init__()
        self.eps = eps
        self.mean = None
        self.std = None

    def forward(self, x, mode='norm'):
        """
        前向传播

        Args:
            x: 输入张量 [B, T, N]
            mode: 'norm' for normalization, 'denorm' for denormalization

        Returns:
            归一化或反归一化后的张量
        """
        if mode == 'norm':
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)
        else:
            raise ValueError(f"Mode must be 'norm' or 'denorm', got {mode}")

    def _normalize(self, x):
        """
        Chronos-2 归一化过程

        Args:
            x: [B, T, N] 输入时间序列

        Returns:
            归一化后的时间序列
        """
        # 计算统计量 (沿时间维度,排除缺失值)
        # 只使用历史数据计算均值和标准差
        self.mean = np.mean(x)  # [B, 1, N]
        self.std = np.sqrt(
            np.var(x) + self.eps
        )  # [B, 1, N]

        # 步骤1: 标准化 z = (x - μ) / σ
        z = (x - self.mean) / self.std

        # 步骤2: arcsinh变换 (log-like transformation)
        # arcsinh(z) = log(z + sqrt(z^2 + 1))
        y = np.asinh(z)

        return y

    def _denormalize(self, y):
        """
        Chronos-2 反归一化过程

        Args:
            y: [B, T, N] 归一化后的预测

        Returns:
            原始尺度的预测
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Must call forward with mode='norm' before denormalizing")

        # 步骤1: 反向arcsinh变换
        # sinh(y) = (e^y - e^(-y)) / 2
        z = np.sinh(y)

        # 步骤2: 反向标准化
        x = z * self.std + self.mean

        return x


class Chronos2MultiVariateScaling(nn.Module):
    """
    Chronos-2 多变量鲁棒缩放

    为多变量时间序列的每个维度分别进行归一化
    """

    def __init__(self, n_vars, eps=1e-6):
        """
        Args:
            n_vars: 变量数量
            eps: 数值稳定性常数
        """
        super(Chronos2MultiVariateScaling, self).__init__()
        self.n_vars = n_vars
        self.eps = eps
        self.scalers = nn.ModuleList([
            Chronos2RobustScaling(eps=eps) for _ in range(n_vars)
        ])

    def forward(self, x, mode='norm'):
        """
        对每个变量分别归一化

        Args:
            x: [T, N] 输入
            mode: 'norm' or 'denorm'

        Returns:
            归一化/反归一化的输出
        """
        outputs = []
        for i in range(self.n_vars):
            # 对每个变量单独处理
            x_i = x[:, i:i + 1]  # [T, 1]
            y_i = self.scalers[i](x_i, mode=mode)
            outputs.append(y_i)

        return np.concatenate(outputs, axis=1)


class Chronos2GroupScaling(nn.Module):
    """
    Chronos-2 Group Scaling

    为不同的组(targets, covariates)使用不同的缩放参数
    """

    def __init__(self, eps=1e-6):
        super(Chronos2GroupScaling, self).__init__()
        self.eps = eps
        # 存储每个组的统计量
        self.group_stats = {}

    def forward(self, x, group_id, mode='norm'):
        """
        Args:
            x: [B, T, 1] 输入 (单个序列)
            group_id: 组标识符
            mode: 'norm' or 'denorm'
        """
        if mode == 'norm':
            return self._normalize(x, group_id)
        else:
            return self._denormalize(x, group_id)

    def _normalize(self, x, group_id):
        # 计算并存储统计量
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps)

        self.group_stats[group_id] = {'mean': mean, 'std': std}

        # 标准化 + arcsinh
        z = (x - mean) / std
        y = torch.asinh(z)

        return y

    def _denormalize(self, y, group_id):
        if group_id not in self.group_stats:
            raise RuntimeError(f"Group {group_id} not found in stats")

        mean = self.group_stats[group_id]['mean']
        std = self.group_stats[group_id]['std']

        z = torch.sinh(y)
        x = z * std + mean

        return x


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    # 测试鲁棒缩放
    print("=" * 80)
    print("测试 Chronos-2 鲁棒缩放")
    print("=" * 80)

    # 创建测试数据 (模拟流量数据,包含极端值)
    batch_size = 4
    seq_len = 96
    n_stations = 10

    # 生成带有极端值的测试数据
    x_test = torch.randn(batch_size, seq_len, n_stations) * 100 + 500
    # 添加一些极端值
    x_test[0, 10:20, 0] *= 5  # 流量突增
    x_test[1, 30:40, 2] *= 0.1  # 流量骤降

    print(f"输入形状: {x_test.shape}")
    print(f"输入统计: min={x_test.min():.2f}, max={x_test.max():.2f}, "
          f"mean={x_test.mean():.2f}, std={x_test.std():.2f}")

    # 1. 测试单变量缩放
    scaler = Chronos2RobustScaling()

    # 归一化
    x_norm = scaler(x_test, mode='norm')
    print(f"\n归一化后统计: min={x_norm.min():.2f}, max={x_norm.max():.2f}, "
          f"mean={x_norm.mean():.2f}, std={x_norm.std():.2f}")

    # 反归一化
    x_denorm = scaler(x_norm, mode='denorm')
    print(f"反归一化后统计: min={x_denorm.min():.2f}, max={x_denorm.max():.2f}, "
          f"mean={x_denorm.mean():.2f}, std={x_denorm.std():.2f}")

    # 检查重建误差
    reconstruction_error = torch.mean(torch.abs(x_test - x_denorm))
    print(f"\n重建误差 (MAE): {reconstruction_error:.6f}")

    # 2. 测试多变量缩放
    print("\n" + "=" * 80)
    print("测试多变量缩放")
    print("=" * 80)

    multi_scaler = Chronos2MultiVariateScaling(n_vars=n_stations)
    x_norm_multi = multi_scaler(x_test, mode='norm')
    x_denorm_multi = multi_scaler(x_norm_multi, mode='denorm')

    reconstruction_error_multi = torch.mean(torch.abs(x_test - x_denorm_multi))
    print(f"多变量重建误差 (MAE): {reconstruction_error_multi:.6f}")

    # 3. 对比标准StandardScaler
    print("\n" + "=" * 80)
    print("对比标准缩放 vs Chronos-2鲁棒缩放")
    print("=" * 80)

    # 标准缩放
    mean_std = torch.mean(x_test, dim=1, keepdim=True)
    std_std = torch.sqrt(torch.var(x_test, dim=1, keepdim=True) + 1e-6)
    x_std_norm = (x_test - mean_std) / std_std

    print(f"标准缩放后: min={x_std_norm.min():.2f}, max={x_std_norm.max():.2f}")
    print(f"Chronos-2缩放后: min={x_norm.min():.2f}, max={x_norm.max():.2f}")
    print("\n✓ Chronos-2的arcsinh变换压缩了极端值的范围!")