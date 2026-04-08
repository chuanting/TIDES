from math import sqrt
import torch
import torch.nn as nn
from transformers import (
    GPT2Config, GPT2Model, GPT2Tokenizer,
    BertConfig, BertModel, BertTokenizer,
    LlamaConfig, LlamaModel, LlamaTokenizer,
    AutoModelForCausalLM, AutoTokenizer, AutoConfig
)
from layers.Embed import PatchEmbedding
from layers.StandardNorm import Normalize
import transformers
import logging
import time

# Set transformer logging to error only
transformers.logging.set_verbosity_error()

# ============================================================================
# Flash Attention 支持检测
# ============================================================================
try:
    from xformers.ops import memory_efficient_attention

    XFORMERS_AVAILABLE = True
    logging.info("✓ xFormers Flash Attention available")
except ImportError:
    XFORMERS_AVAILABLE = False
    logging.warning("✗ xFormers not available, falling back to standard attention")

# PyTorch 2.0+ Scaled Dot Product Attention (自带Flash Attention优化)
PYTORCH_SDPA_AVAILABLE = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
if PYTORCH_SDPA_AVAILABLE:
    logging.info("✓ PyTorch SDPA (built-in Flash Attention) available")


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


# ============================================================================
# 优化版本1: ReprogrammingLayer with Flash Attention
# ============================================================================
class ReprogrammingLayer(nn.Module):
    """
    优化的Reprogramming层，支持Flash Attention加速
    """

    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.d_keys = d_keys
        self.dropout = nn.Dropout(attention_dropout)
        self.attention_dropout = attention_dropout

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        # 投影 Q, K, V
        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        # 使用优化的注意力计算
        out = self.reprogramming_flash(target_embedding, source_embedding, value_embedding)
        out = out.reshape(B, L, -1)
        return self.out_projection(out)

    def reprogramming_flash(self, target_embedding, source_embedding, value_embedding):
        """
        使用Flash Attention优化的reprogramming计算
        """
        B, L, H, E = target_embedding.shape
        S = source_embedding.shape[0]

        # 重塑为标准的多头注意力格式
        # Q: [B, L, H, E] -> [B, H, L, E]
        # K: [S, H, E] -> [1, H, S, E] (broadcast batch dimension)
        # V: [S, H, E] -> [1, H, S, E]
        q = target_embedding.transpose(1, 2)  # [B, H, L, E]
        k = source_embedding.unsqueeze(0).transpose(1, 2)  # [1, H, S, E]
        v = value_embedding.unsqueeze(0).transpose(1, 2)  # [1, H, S, E]

        # 广播K和V到batch维度
        k = k.expand(B, -1, -1, -1)  # [B, H, S, E]
        v = v.expand(B, -1, -1, -1)  # [B, H, S, E]

        # 方案1: 使用PyTorch 2.0+ SDPA (推荐)
        if PYTORCH_SDPA_AVAILABLE:
            # PyTorch的SDPA会自动选择最优实现（包括Flash Attention）
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=False,
                scale=None  # 自动使用 1/sqrt(d_k)
            )
        # 方案2: 使用xFormers (如果PyTorch SDPA不可用)
        elif XFORMERS_AVAILABLE:
            # xFormers期望的格式: [B, L, H, E]
            q_xf = q.transpose(1, 2)  # [B, L, H, E]
            k_xf = k.transpose(1, 2)  # [B, S, H, E]
            v_xf = v.transpose(1, 2)  # [B, S, H, E]

            attn_output = memory_efficient_attention(
                q_xf, k_xf, v_xf,
                attn_bias=None,
                p=self.attention_dropout if self.training else 0.0,
                scale=1.0 / sqrt(E)
            )
            attn_output = attn_output.transpose(1, 2)  # [B, H, L, E]
        # 后备方案: 标准注意力
        else:
            scale = 1.0 / sqrt(E)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, L, S]
            attn_probs = torch.softmax(scores, dim=-1)
            attn_probs = self.dropout(attn_probs)
            attn_output = torch.matmul(attn_probs, v)  # [B, H, L, E]

        # 转回原始格式 [B, H, L, E] -> [B, L, H, E]
        reprogramming_embedding = attn_output.transpose(1, 2)
        return reprogramming_embedding


# ============================================================================
# 优化版本2: SpatialAttentionLayer with Flash Attention
# ============================================================================
class SpatialAttentionLayer(nn.Module):
    """
    增强的空间注意力机制，支持Flash Attention加速
    用于捕获基站之间的关系
    """

    def __init__(self, in_channels, num_heads=4, dropout=0.1):
        super(SpatialAttentionLayer, self).__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        assert self.head_dim * num_heads == in_channels, "in_channels must be divisible by num_heads"

        # Projections for Q, K, V
        self.query_proj = nn.Linear(in_channels, in_channels)
        self.key_proj = nn.Linear(in_channels, in_channels)
        self.value_proj = nn.Linear(in_channels, in_channels)

        self.out_proj = nn.Linear(in_channels, in_channels)
        self.dropout = nn.Dropout(dropout)
        self.dropout_p = dropout

    def forward(self, x, adj_matrix=None, mask=None):
        """
        Args:
            x: Input tensor [B, N, C] where N is number of stations, C is channels
            adj_matrix: Optional adjacency matrix [N, N] to inform attention
            mask: Optional mask tensor [B, N] or [N]

        Returns:
            Updated tensor with spatial attention applied
        """
        batch_size, num_stations, _ = x.size()

        # Projection and reshape
        q = self.query_proj(x).view(batch_size, num_stations, self.num_heads, self.head_dim)
        k = self.key_proj(x).view(batch_size, num_stations, self.num_heads, self.head_dim)
        v = self.value_proj(x).view(batch_size, num_stations, self.num_heads, self.head_dim)

        # 使用Flash Attention优化的注意力计算
        context, attn_probs = self._compute_attention_flash(
            q, k, v, adj_matrix, mask, batch_size, num_stations
        )

        # Output projection
        output = self.out_proj(context)

        return output, attn_probs

    def _compute_attention_flash(self, q, k, v, adj_matrix, mask, batch_size, num_stations):
        """
        使用Flash Attention优化的注意力计算
        """
        # 方案1: 使用PyTorch 2.0+ SDPA (推荐，自动处理mask和bias)
        if PYTORCH_SDPA_AVAILABLE and adj_matrix is None and mask is None:
            # 转换为 [B, H, N, D] 格式
            q = q.transpose(1, 2)  # [B, H, N, D]
            k = k.transpose(1, 2)  # [B, H, N, D]
            v = v.transpose(1, 2)  # [B, H, N, D]

            # PyTorch SDPA会自动使用Flash Attention（如果可用）
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False
            )

            # [B, H, N, D] -> [B, N, H, D] -> [B, N, C]
            context = attn_output.transpose(1, 2).contiguous().view(batch_size, num_stations, -1)
            attn_probs = None  # SDPA不返回attention weights

        # 方案2: 使用xFormers (当有adjacency matrix或mask时)
        elif XFORMERS_AVAILABLE and (adj_matrix is None and mask is None):
            # xFormers期望 [B, N, H, D] 格式
            attn_output = memory_efficient_attention(
                q, k, v,
                attn_bias=None,
                p=self.dropout_p if self.training else 0.0
            )
            context = attn_output.contiguous().view(batch_size, num_stations, -1)
            attn_probs = None

        # 后备方案: 标准注意力 (当需要adjacency matrix或mask时)
        else:
            # Transpose for attention computation
            q = q.transpose(1, 2)  # [B, H, N, D]
            k = k.transpose(1, 2)  # [B, H, N, D]
            v = v.transpose(1, 2)  # [B, H, N, D]

            # Compute attention scores
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, N, N]

            # Incorporate adjacency information if provided
            if adj_matrix is not None:
                adj_bias = adj_matrix.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
                adj_bias = (1.0 - adj_bias) * -10000.0
                attn_scores = attn_scores + adj_bias

            # Apply mask if provided
            if mask is not None:
                if mask.dim() == 1:
                    mask = mask.unsqueeze(0)  # [1, N]
                mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
                attn_scores = attn_scores.masked_fill(mask == 0, -10000.0)

            # Softmax attention
            attn_probs = torch.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)

            # Apply attention
            context = torch.matmul(attn_probs, v)  # [B, H, N, D]
            context = context.transpose(1, 2).contiguous()  # [B, N, H, D]
            context = context.view(batch_size, num_stations, -1)  # [B, N, C]

        return context, attn_probs


# ============================================================================
# 主模型
# ============================================================================
class Model(nn.Module):
    def __init__(self, configs, patch_len=None, stride=None):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len if patch_len is None else patch_len
        self.stride = configs.stride if stride is None else stride
        self.neighbor = configs.neighbor
        self.use_spatial_attn = True
        self.d_model = configs.d_model
        self.enhanced_prompt = configs.enhanced_prompt

        # Performance profiling
        self.timers = {}
        self.profiling = True

        # 添加 LLM 类型标识（在 _init_llm_model 之前）
        self.llm_type = configs.llm_model.upper()  # 添加这一行

        # 优化标志
        self.use_flash_attention = PYTORCH_SDPA_AVAILABLE or XFORMERS_AVAILABLE
        if self.use_flash_attention:
            logging.info("✓ Flash Attention enabled for DSTraffic model")
        else:
            logging.warning("✗ Flash Attention not available, using standard attention")

        # Initialize LLM model based on config
        self._init_llm_model(configs)

        self.dropout = nn.Dropout(configs.dropout)
        self.patch_embedding = PatchEmbedding(configs.d_model, self.patch_len, self.stride, configs.dropout)

        # Get word embeddings from LLM
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        # 使用优化版本的ReprogrammingLayer
        self.reprogramming_layer = ReprogrammingLayer(
            configs.d_model, configs.n_heads, self.d_ff, self.d_llm
        )

        # 使用优化版本的SpatialAttentionLayer
        if self.use_spatial_attn:
            self.spatial_attn = SpatialAttentionLayer(
                in_channels=configs.d_model,
                num_heads=configs.n_heads,
                dropout=configs.dropout
            )

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        self.output_projection = FlattenHead(
            configs.enc_in,
            self.head_nf,
            self.pred_len,
            head_dropout=configs.dropout
        )

        # Dataset description
        self.description = 'This dataset denotes the wireless traffic of base stations of a city.'
        if hasattr(configs, 'prompt_domain') and configs.prompt_domain and hasattr(configs, 'content'):
            self.description = configs.content

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def _init_llm_model(self, configs):
        """
        初始化LLM模型，并启用Flash Attention优化
        """
        llm_model_type = configs.llm_model.upper()
        self.llm_type = llm_model_type  # 保存类型
        self.use_mixed_precision = True  # Enable mixed precision by default
        # 标记是否为encoder-decoder模型
        self.is_encoder_decoder = False

        # 🔥 通用的Flash Attention配置
        flash_attn_config = {
            'attn_implementation': 'sdpa'  # 使用PyTorch的scaled_dot_product_attention
        }

        if llm_model_type == 'GPT2':
            self._init_gpt2(configs, flash_attn_config)
        elif llm_model_type == 'BERT':
            self._init_bert(configs, flash_attn_config)
        elif llm_model_type == 'LLAMA':
            self._init_llama(configs, flash_attn_config)
        elif llm_model_type == 'DEEPSEEK':
            self._init_deepseek(configs, flash_attn_config)
        else:
            raise ValueError(f"Unsupported LLM model type: {configs.llm_model}")

        # Ensure padding token is set
        if not hasattr(self.tokenizer, 'pad_token') or self.tokenizer.pad_token is None:
            if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                pad_token = '[PAD]'
                self.tokenizer.add_special_tokens({'pad_token': pad_token})
                self.tokenizer.pad_token = pad_token

        # 🔥 为LLM启用梯度检查点（节省显存）
        if hasattr(self.llm_model, 'gradient_checkpointing_enable'):
            self.llm_model.gradient_checkpointing_enable()
            logging.info("✓ Gradient checkpointing enabled for LLM")

        # Freeze LLM parameters
        for param in self.llm_model.parameters():
            param.requires_grad = False

        logging.info(f"✓ LLM model initialized: {llm_model_type}")

    def _init_gpt2(self, configs, flash_attn_config):
        """初始化GPT2模型"""
        try:
            gpt2_config = GPT2Config.from_pretrained('gpt2')
            gpt2_config.num_hidden_layers = configs.llm_layers
            gpt2_config.output_attentions = True
            gpt2_config.output_hidden_states = True

            # 🔥 启用Flash Attention (PyTorch 2.0+)
            if PYTORCH_SDPA_AVAILABLE:
                gpt2_config._attn_implementation = 'sdpa'

            self.llm_model = GPT2Model.from_pretrained(
                'gpt2',
                trust_remote_code=True,
                local_files_only=False,
                config=gpt2_config,
            )
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                'gpt2',
                trust_remote_code=True,
                local_files_only=False,
            )
        except Exception as e:
            logging.warning(f"Failed to load GPT2: {e}")
            raise

    def _init_bert(self, configs, flash_attn_config):
        """初始化BERT模型"""
        try:
            bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')
            bert_config.num_hidden_layers = configs.llm_layers
            bert_config.output_attentions = True
            bert_config.output_hidden_states = True

            # 🔥 启用Flash Attention
            if PYTORCH_SDPA_AVAILABLE:
                bert_config._attn_implementation = 'sdpa'

            self.llm_model = BertModel.from_pretrained(
                'google-bert/bert-base-uncased',
                trust_remote_code=True,
                local_files_only=False,
                config=bert_config,
            )
            self.tokenizer = BertTokenizer.from_pretrained(
                'google-bert/bert-base-uncased',
                trust_remote_code=True,
                local_files_only=False,
            )
        except Exception as e:
            logging.warning(f"Failed to load BERT: {e}")
            raise

    def _init_llama(self, configs, flash_attn_config):
        """初始化LLAMA模型"""
        model_path = "./llama-7b"
        try:
            llama_config = LlamaConfig.from_pretrained(model_path, local_files_only=True)
            llama_config.num_hidden_layers = configs.llm_layers
            llama_config.output_attentions = True
            llama_config.output_hidden_states = True

            # 🔥 启用Flash Attention
            if PYTORCH_SDPA_AVAILABLE:
                llama_config._attn_implementation = 'sdpa'

            self.llm_model = LlamaModel.from_pretrained(
                model_path,
                local_files_only=True,
                config=llama_config,
                trust_remote_code=True,
            )
            self.tokenizer = LlamaTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True
            )
        except Exception as e:
            logging.warning(f"Failed to load LLAMA locally: {e}")
            raise

    def _init_deepseek(self, configs, flash_attn_config):
        """初始化DeepSeek模型"""
        model_path = "./ds_llama_8b/"
        try:
            ds_config = AutoConfig.from_pretrained(
                model_path, local_files_only=True, trust_remote_code=True
            )
            ds_config.num_hidden_layers = configs.llm_layers
            ds_config.output_attentions = True
            ds_config.output_hidden_states = True

            # 🔥 启用Flash Attention
            if PYTORCH_SDPA_AVAILABLE:
                ds_config._attn_implementation = 'sdpa'

            self.llm_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                config=ds_config,
                trust_remote_code=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True,
                use_fast=False
            )
        except Exception as e:
            logging.warning(f"Failed to load DeepSeek: {e}")
            raise

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """Forward pass for the DSTraffic model"""
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """Optimized forecasting method for DSTraffic model"""
        # Early exit for empty or zero inputs
        if x_enc.size(0) == 0 or torch.all(x_enc == 0):
            return torch.zeros((x_enc.size(0), self.pred_len, 1), device=x_enc.device)

        # Convert to bfloat16 for mixed precision if supported
        dtype = torch.bfloat16 if self.use_mixed_precision and torch.cuda.is_available() else torch.float32

        # Keep original shape for reference
        B, T, N = x_enc.size()

        # Normalize input
        x_enc_norm = self.normalize_layers(x_enc, 'norm')

        # Reshape for statistics calculation
        x_reshaped = x_enc_norm.permute(0, 2, 1).reshape(B * N, T, 1).contiguous()

        # Calculate enhanced statistics
        stats_dict = self._calculate_enhanced_statistics(x_reshaped)

        # Generate prompts
        prompts = self._batch_generate_enhanced_prompts(stats_dict, B * N)

        # Tokenize prompts
        tokenize_kwargs = {
            'return_tensors': 'pt',
            'padding': True,
            'truncation': True,
            'max_length': 2048
        }

        prompt_tokens = self.tokenizer(prompts, **tokenize_kwargs).input_ids.to(x_enc.device)

        # Get prompt embeddings
        with torch.no_grad():
            prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_tokens)

        # Map word embeddings
        source_embeddings = self.mapping_layer(
            self.word_embeddings.permute(1, 0)
        ).permute(1, 0)
        if self.use_mixed_precision:
            source_embeddings = source_embeddings.to(dtype)

        # Apply patch embedding
        patch_input = x_enc_norm.permute(0, 2, 1).contiguous()
        if self.use_mixed_precision:
            patch_input = patch_input.to(dtype)

        enc_out, n_vars = self.patch_embedding(patch_input)

        # Apply spatial attention with Flash Attention optimization
        if self.use_spatial_attn and hasattr(self, 'spatial_attn'):
            adj_matrix = getattr(self, 'adj', None)
            if adj_matrix is not None and adj_matrix.device != x_enc.device:
                adj_matrix = adj_matrix.to(x_enc.device)

            spatial_out, _ = self.spatial_attn(enc_out, adj_matrix)
            enc_out = spatial_out.contiguous().view(B * N, -1, enc_out.size(-1))

        # Apply reprogramming layer with Flash Attention optimization
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        # Concatenate prompt embeddings with encoded input
        llm_encoder_input = torch.cat([prompt_embeddings, enc_out], dim=1)

        # ==================== 其他模型（GPT-2, BERT 等）====================
        # Pass through LLM model with mixed precision
        if self.use_mixed_precision and torch.cuda.is_available():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                llm_output = self.llm_model(inputs_embeds=llm_encoder_input)
        else:
            llm_output = self.llm_model(inputs_embeds=llm_encoder_input)

        # Handle different output structures
        if hasattr(llm_output, 'last_hidden_state'):
            dec_out = llm_output.last_hidden_state
        elif hasattr(llm_output, 'hidden_states') and llm_output.hidden_states is not None:
            dec_out = llm_output.hidden_states[-1]
        else:
            logging.warning("Unable to find hidden states. Using logits instead.")
            dec_out = llm_output.logits

        # Extract necessary features and reshape
        dec_out = dec_out.narrow(2, 0, self.d_ff)
        dec_out = dec_out.view(B, N, dec_out.shape[-2], dec_out.shape[-1])
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        # Apply output projection
        if self.patch_nums > 0:
            output_input = dec_out[:, :, :, -self.patch_nums:]
            if self.use_mixed_precision:
                output_input = output_input.to(dtype)
            dec_out = self.output_projection(output_input)
        else:
            dec_out = self.output_projection(dec_out[:, :, :, :])

        # Final reshaping
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        # Denormalize
        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    def calcute_lags(self, x_enc):
        """Calculate autocorrelation lags using FFT"""
        x_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = x_fft * torch.conj(x_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags

    # ========================================================================
    # 辅助方法 (需要从原文件复制)
    # ========================================================================
    def _calculate_enhanced_statistics(self, x_reshaped):
        """Calculate enhanced statistics for traffic data

        Args:
            x_reshaped: Reshaped input tensor [B*N, T, 1]

        Returns:
            Dictionary of enhanced traffic statistics
        """
        # Basic statistics (already in your code)
        min_values = torch.min(x_reshaped, dim=1)[0]
        max_values = torch.max(x_reshaped, dim=1)[0]
        medians = torch.median(x_reshaped, dim=1).values
        means = torch.mean(x_reshaped, dim=1)

        # Calculate lags efficiently using FFT
        x_fft = torch.fft.rfft(x_reshaped.permute(0, 2, 1).contiguous(), dim=-1)
        corr = torch.fft.irfft(x_fft * torch.conj(x_fft), dim=-1)
        mean_corr = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_corr, self.top_k, dim=-1)

        # Calculate trends
        trends = torch.sum(torch.diff(x_reshaped, dim=1), dim=1).squeeze(-1)

        # NEW TRAFFIC-SPECIFIC STATISTICS

        # 1. Peak-to-Average Ratio (PAR)
        par_values = max_values.squeeze(-1) / (means.squeeze(-1) + 1e-6)

        # 2. Time-of-day patterns (assuming 24-hour data with 15-min intervals)
        # This assumes your sequence length is at least 96 (24 hours with 15-min intervals)
        # Adjust the calculations based on your actual data frequency
        seq_len = x_reshaped.shape[1]
        points_per_day = min(seq_len, 96)  # 96 points for a day with 15-min intervals

        # Reshape to organize by time of day
        if seq_len >= points_per_day:
            daily_data = x_reshaped[:, -points_per_day:, 0]

            # Morning peak (7-10 AM: points 28-40 for 15-min intervals)
            morning_indices = list(range(28, min(40, points_per_day)))
            morning_avg = torch.mean(daily_data[:, morning_indices], dim=1)

            # Evening peak (4-7 PM: points 64-76 for 15-min intervals)
            evening_indices = list(range(64, min(76, points_per_day)))
            evening_avg = torch.mean(daily_data[:, evening_indices], dim=1)

            # Night/off-peak (11 PM-5 AM: points 84-96 and 0-20)
            night_indices = list(range(84, points_per_day)) + list(range(0, min(20, points_per_day)))
            night_avg = torch.mean(daily_data[:, [i % points_per_day for i in night_indices]], dim=1)

            # Calculate rush hour intensity
            non_rush_indices = list(set(range(points_per_day)) - set(morning_indices) - set(evening_indices))
            non_rush_avg = torch.mean(daily_data[:, non_rush_indices], dim=1)

            # Rush hour intensity: ratio of rush hour to non-rush hour traffic
            rush_intensity = (morning_avg + evening_avg) / (2 * non_rush_avg + 1e-6)

            # Morning vs evening pattern
            morning_evening_ratio = morning_avg / (evening_avg + 1e-6)
        else:
            # If we don't have enough data for a full day, use zeros as placeholders
            morning_avg = torch.zeros_like(means.squeeze(-1))
            evening_avg = torch.zeros_like(means.squeeze(-1))
            night_avg = torch.zeros_like(means.squeeze(-1))
            rush_intensity = torch.ones_like(means.squeeze(-1))
            morning_evening_ratio = torch.ones_like(means.squeeze(-1))

        # 3. Burstiness - Coefficient of variation (standard deviation / mean)
        stdevs = torch.std(x_reshaped, dim=1).squeeze(-1)
        burstiness = stdevs / (means.squeeze(-1) + 1e-6)

        # 4. Volatility - calculate using rolling standard deviation (simplified version)
        half_window = min(6, seq_len // 2)  # 1.5 hours with 15-min data
        if seq_len > half_window * 2:
            first_half = torch.std(x_reshaped[:, :half_window, 0], dim=1)
            second_half = torch.std(x_reshaped[:, -half_window:, 0], dim=1)
            volatility_change = second_half / (first_half + 1e-6)
        else:
            volatility_change = torch.ones_like(means.squeeze(-1))

        return {
            'min_values': min_values.squeeze(-1),
            'max_values': max_values.squeeze(-1),
            'medians': medians.squeeze(-1),
            'trends': trends,
            'lags': lags,
            # New statistics
            'par': par_values,
            'morning_avg': morning_avg,
            'evening_avg': evening_avg,
            'night_avg': night_avg,
            'rush_intensity': rush_intensity,
            'morning_evening_ratio': morning_evening_ratio,
            'burstiness': burstiness,
            'volatility_change': volatility_change
        }

    def _batch_generate_enhanced_prompts(self, stats_dict, batch_size):
        """
        Generate prompts with enhanced traffic statistics for the entire batch

        Args:
            stats_dict: Dictionary containing all traffic statistics
            batch_size: Size of the batch

        Returns:
            List of formatted prompts
        """
        # Pre-format static parts of the prompt
        static_prompt_part = (
            f"<|start_prompt|>Dataset description: {self.description}"
            f"Task description: forecast the next {self.pred_len} steps given the previous {self.seq_len} steps information; "
            f"Input statistics: "
        )

        # Create prompt for each batch item efficiently
        prompts = []
        for i in range(batch_size):
            trend_str = 'upward' if stats_dict['trends'][i] > 0 else 'downward'

            # Format basic statistics
            basic_stats = (
                f"min value {stats_dict['min_values'][i].item():.4f}, "
                f"max value {stats_dict['max_values'][i].item():.4f}, "
                f"median value {stats_dict['medians'][i].item():.4f}, "
                f"trend direction is {trend_str}, "
                f"top 5 lags are : {stats_dict['lags'][i].tolist()}, "
            )

            # Format enhanced traffic-specific statistics
            traffic_stats = (
                f"peak-to-average ratio: {stats_dict['par'][i].item():.4f}, "
                f"rush hour intensity: {stats_dict['rush_intensity'][i].item():.4f}, "
                f"morning-to-evening ratio: {stats_dict['morning_evening_ratio'][i].item():.4f}, "
                f"traffic burstiness: {stats_dict['burstiness'][i].item():.4f}, "
                f"recent volatility change: {stats_dict['volatility_change'][i].item():.4f}"
            )

            if self.enhanced_prompt:
                prompt = f"{static_prompt_part}{basic_stats}{traffic_stats}<|<end_prompt>|>"
            else:
                prompt = f"{static_prompt_part}{basic_stats}<|<end_prompt>|>"
            prompts.append(prompt)

        return prompts
