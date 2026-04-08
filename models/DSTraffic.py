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


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)
        out = out.reshape(B, L, -1)
        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape
        scale = 1. / sqrt(E)
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)
        return reprogramming_embedding


class SpatialAttentionLayer(nn.Module):
    """
    Enhanced spatial attention mechanism for capturing relationships between base stations
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
        q = self.query_proj(x).view(batch_size, num_stations, self.num_heads, -1)
        k = self.key_proj(x).view(batch_size, num_stations, self.num_heads, -1)
        v = self.value_proj(x).view(batch_size, num_stations, self.num_heads, -1)

        # Transpose for attention computation
        q = q.transpose(1, 2)  # [B, H, N, D]
        k = k.transpose(1, 2)  # [B, H, N, D]
        v = v.transpose(1, 2)  # [B, H, N, D]

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, N, N]

        # Incorporate adjacency information if provided
        if adj_matrix is not None:
            # Reshape adjacency matrix for broadcasting
            adj_bias = adj_matrix.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]

            # Add as a bias term (negative values for non-connected nodes)
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

        # Output projection
        output = self.out_proj(context)

        return output, attn_probs


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
        self.use_spatial_attn = True  # New flag to enable spatial attention
        self.d_model = configs.d_model
        self.enhanced_prompt = configs.enhanced_prompt

        # Performance profiling
        self.timers = {}
        self.profiling = True  # Set to True to enable detailed profiling

        # Initialize LLM model based on config
        self._init_llm_model(configs)

        self.dropout = nn.Dropout(configs.dropout)
        self.patch_embedding = PatchEmbedding(configs.d_model, self.patch_len, self.stride, configs.dropout)

        # Get word embeddings from LLM
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        # Reprogram LLM for time series forecasting
        self.reprogramming_layer = ReprogrammingLayer(
            configs.d_model, configs.n_heads, self.d_ff, self.d_llm
        )

        # New spatial attention layer
        if self.use_spatial_attn:
            self.spatial_attn = SpatialAttentionLayer(
                in_channels=configs.d_model,
                num_heads=configs.n_heads,
                dropout=configs.dropout
            )

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        self.output_projection = FlattenHead(
                configs.neighbor, self.head_nf, self.pred_len, head_dropout=configs.dropout
            )

        self.normalize_layers = Normalize(configs.neighbor, affine=False)

        # Dataset description
        self.description = 'This dataset denotes the wireless traffic of base stations of a city.'
        if hasattr(configs, 'prompt_domain') and configs.prompt_domain and hasattr(configs, 'content'):
            self.description = configs.content

        # Track if we're using mixed precision
        self.use_mixed_precision = True

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

    @torch.no_grad()  # Ensure we don't track gradients for efficiency
    def _calculate_statistics(self, x_reshaped):
        """Calculate statistics for input data efficiently

        Args:
            x_reshaped: Reshaped input tensor [B*N, T, 1]

        Returns:
            Tuple of statistics tensors
        """
        # Calculate all statistics at once
        min_values = torch.min(x_reshaped, dim=1)[0]
        max_values = torch.max(x_reshaped, dim=1)[0]
        medians = torch.median(x_reshaped, dim=1).values

        # More efficient lag calculation using FFT
        x_fft = torch.fft.rfft(x_reshaped.permute(0, 2, 1).contiguous(), dim=-1)
        corr = torch.fft.irfft(x_fft * torch.conj(x_fft), dim=-1)
        mean_corr = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_corr, self.top_k, dim=-1)

        # Calculate trends (more efficiently)
        trends = torch.sum(torch.diff(x_reshaped, dim=1), dim=1).squeeze(-1)

        return min_values, max_values, medians, lags, trends

    def _init_llm_model(self, configs):
        """Initialize the LLM model based on config"""
        llm_type = configs.llm_model
        num_layers = configs.llm_layers

        # Common config settings
        config_args = {
            'num_hidden_layers': num_layers,
            'output_attentions': True,
            'output_hidden_states': True
        }

        if llm_type == 'GPT2':
            self._init_gpt2_model(config_args)
        elif llm_type == 'BERT':
            self._init_bert_model(config_args)
        elif llm_type == 'LLAMA':
            self._init_llama_model(config_args)
        elif llm_type == 'deepseek':
            self._init_deepseek_model(config_args)
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")

        # Ensure padding token is set
        if not hasattr(self.tokenizer, 'pad_token') or self.tokenizer.pad_token is None:
            if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                pad_token = '[PAD]'
                self.tokenizer.add_special_tokens({'pad_token': pad_token})
                self.tokenizer.pad_token = pad_token

        # Freeze LLM parameters
        for param in self.llm_model.parameters():
            param.requires_grad = False

    def _init_gpt2_model(self, config_args):
        """Initialize GPT2 model"""
        try:
            model_name = 'openai-community/gpt2'
            self.gpt2_config = GPT2Config.from_pretrained(model_name, **config_args)
            self.llm_model = GPT2Model.from_pretrained(
                model_name, config=self.gpt2_config, local_files_only=True, trust_remote_code=True
            )
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                model_name, local_files_only=True, trust_remote_code=True
            )
        except Exception as e:
            logging.warning(f"Failed to load models locally: {e}. Downloading from HuggingFace.")
            model_name = 'openai-community/gpt2'
            self.gpt2_config = GPT2Config.from_pretrained(model_name, **config_args)
            self.llm_model = GPT2Model.from_pretrained(
                model_name, config=self.gpt2_config, trust_remote_code=True
            )
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )

    def _init_bert_model(self, config_args):
        """Initialize BERT model"""
        try:
            model_name = 'google-bert/bert-base-uncased'
            self.bert_config = BertConfig.from_pretrained(model_name, **config_args)
            self.llm_model = BertModel.from_pretrained(
                model_name, config=self.bert_config, local_files_only=True, trust_remote_code=True
            )
            self.tokenizer = BertTokenizer.from_pretrained(
                model_name, local_files_only=True, trust_remote_code=True
            )
        except Exception as e:
            logging.warning(f"Failed to load models locally: {e}. Downloading from HuggingFace.")
            model_name = 'google-bert/bert-base-uncased'
            self.bert_config = BertConfig.from_pretrained(model_name, **config_args)
            self.llm_model = BertModel.from_pretrained(
                model_name, config=self.bert_config, trust_remote_code=True
            )
            self.tokenizer = BertTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )

    def _init_llama_model(self, config_args):
        """Initialize LLAMA model"""
        try:
            model_name = 'huggyllama/llama-7b'
            self.llama_config = LlamaConfig.from_pretrained(model_name, **config_args)
            self.llm_model = LlamaModel.from_pretrained(
                model_name, config=self.llama_config, local_files_only=True, trust_remote_code=True
            )
            self.tokenizer = LlamaTokenizer.from_pretrained(
                model_name, local_files_only=True, trust_remote_code=True
            )
        except Exception as e:
            logging.warning(f"Failed to load models locally: {e}. Downloading from HuggingFace.")
            model_name = 'huggyllama/llama-7b'
            self.llama_config = LlamaConfig.from_pretrained(model_name, **config_args)
            self.llm_model = LlamaModel.from_pretrained(
                model_name, config=self.llama_config, trust_remote_code=True
            )
            self.tokenizer = LlamaTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )

    def _init_deepseek_model(self, config_args):
        """Initialize DeepSeek model"""
        try:
            model_path = "./ds_llama_8b/"
            self.ds_config = AutoConfig.from_pretrained(model_path, **config_args)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                model_path, config=self.ds_config, local_files_only=True, trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, local_files_only=True, trust_remote_code=True, use_fast=False
            )
        except Exception as e:
            logging.warning(f"Failed to load DeepSeek model locally: {e}")
            # Since this is a local custom model, we don't try to download from HF
            raise ValueError(
                "DeepSeek model files not found locally. Please make sure the files exist in ./ds_llama_8b/")

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Forward pass for the DSTraffic model

        Args:
            x_enc: Input sequence (B, T, N)
            x_mark_enc: Time features for input sequence
            x_dec: Decoder input sequence (not used in this model)
            x_mark_dec: Time features for output sequence (not used in this model)
            mask: Optional mask (not used in this model)

        Returns:
            Model output predictions for the requested prediction length
        """
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        Optimized forecasting method for DSTraffic model

        Args:
            x_enc: Encoder input (B, T, N)
            x_mark_enc: Time features for encoder (not used)
            x_dec: Decoder input (not used)
            x_mark_dec: Time features for decoder (not used)

        Returns:
            Forecasted time series
        """
        # Use profiling if enabled

        # Early exit for empty or zero inputs
        if x_enc.size(0) == 0 or torch.all(x_enc == 0):
            return torch.zeros((x_enc.size(0), self.pred_len, 1), device=x_enc.device)

        # Convert to bfloat16 for mixed precision if supported
        dtype = torch.bfloat16 if self.use_mixed_precision and torch.cuda.is_available() else torch.float32

        # Keep original shape for reference
        B, T, N = x_enc.size()

        # Normalize input (without changing shape)
        x_enc_norm = self.normalize_layers(x_enc, 'norm')

        # Reshape for statistics calculation (more efficiently with contiguous memory)
        x_reshaped = x_enc_norm.permute(0, 2, 1).reshape(B * N, T, 1).contiguous()

        # Calculate enhanced statistics
        stats_dict = self._calculate_enhanced_statistics(x_reshaped)

        # Generate prompts efficiently using enhanced batch method
        prompts = self._batch_generate_enhanced_prompts(stats_dict, B * N)

        # Tokenize prompts (using efficient batching)
        tokenize_kwargs = {
            'return_tensors': 'pt',
            'padding': True,
            'truncation': True,
            'max_length': 2048
        }

        prompt_tokens = self.tokenizer(prompts, **tokenize_kwargs).input_ids.to(x_enc.device)

        # Get prompt embeddings
        # self._start_timer("embed_prompts")
        with torch.no_grad():  # No need to track gradients for embedding lookup
            prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_tokens)
        # self._stop_timer("embed_prompts")

        # Map word embeddings (compute once and cache if possible)
        source_embeddings = self.mapping_layer(
                self.word_embeddings.permute(1, 0)
            ).permute(1, 0)
        if self.use_mixed_precision:
            source_embeddings = source_embeddings.to(dtype)
        # Apply patch embedding (use x_enc_norm to avoid reshaping again)
        # We need to convert to bfloat16 for mixed precision
        patch_input = x_enc_norm.permute(0, 2, 1).contiguous()
        if self.use_mixed_precision:
            patch_input = patch_input.to(dtype)

        enc_out, n_vars = self.patch_embedding(patch_input)

        # Apply spatial attention if enabled
        if self.use_spatial_attn and hasattr(self, 'spatial_attn'):
            # self._start_timer("spatial_attention")
            # Reshape for spatial attention (B*N, T, C) -> (B, N, T*C)
            # spatial_input = enc_out.contiguous().view(B, N, -1)

            # Apply spatial attention using adjacency matrix if available
            adj_matrix = getattr(self, 'adj_k', None)
            if adj_matrix is not None and adj_matrix.device != x_enc.device:
                adj_matrix = adj_matrix.to(x_enc.device)

            # spatial_out, _ = self.spatial_attn(spatial_input, adj_matrix)
            spatial_out, _ = self.spatial_attn(enc_out, adj_matrix)

            # Reshape back to original format (B, N, T*C) -> (B*N, T, C)
            enc_out = spatial_out.contiguous().view(B * N, -1, enc_out.size(-1))

        # Apply reprogramming layer
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        # Concatenate prompt embeddings with encoded input
        llm_input = torch.cat([prompt_embeddings, enc_out], dim=1)

        # Pass through LLM model with mixed precision
        if self.use_mixed_precision and torch.cuda.is_available():
            # Enable gradient checkpointing during training to save memory
            if self.training and hasattr(self.llm_model, "gradient_checkpointing_enable"):
                self.llm_model.gradient_checkpointing_enable()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                llm_output = self.llm_model(inputs_embeds=llm_input)
        else:
            llm_output = self.llm_model(inputs_embeds=llm_input)

        # Handle different output structures based on model type
        if hasattr(llm_output, 'last_hidden_state'):
            dec_out = llm_output.last_hidden_state
        elif hasattr(llm_output, 'hidden_states') and llm_output.hidden_states is not None:
            # Use the last layer's hidden states
            dec_out = llm_output.hidden_states[-1]
        else:
            # For CausalLM models that don't provide hidden states directly
            print("Warning: Unable to find hidden states. Using logits instead.")
            dec_out = llm_output.logits

        # Extract necessary features and reshape efficiently
        # Use narrow instead of slice when possible for memory efficiency
        dec_out = dec_out.narrow(2, 0, self.d_ff)

        # Reshape for output projection efficiently by using view where possible
        dec_out = dec_out.view(B, N, dec_out.shape[-2], dec_out.shape[-1])
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        # Apply output projection (only to the necessary part of the tensor)
        if self.patch_nums > 0:
            output_input = dec_out[:, :, :, -self.patch_nums:]
            if self.use_mixed_precision:
                output_input = output_input.to(dtype)
            dec_out = self.output_projection(output_input)
        else:
            # Handle edge case where patch_nums is 0
            dec_out = self.output_projection(dec_out[:, :, :, :])

        # Final reshaping
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        # Denormalize
        dec_out = self.normalize_layers(dec_out, 'denorm')

        # Clean up large temporary tensors to reduce memory usage
        del prompt_embeddings, llm_input, llm_output

        # Return only first feature dimension
        return dec_out

    def calcute_lags(self, x_enc):
        """
        Calculate autocorrelation lags using FFT

        Args:
            x_enc: Input time series data

        Returns:
            Top-k lag indices based on autocorrelation
        """
        # Transform to frequency domain
        x_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)

        # Calculate autocorrelation (using FFT method)
        res = x_fft * torch.conj(x_fft)
        corr = torch.fft.irfft(res, dim=-1)

        # Get mean correlation across channels
        mean_value = torch.mean(corr, dim=1)

        # Find top-k lags
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags
