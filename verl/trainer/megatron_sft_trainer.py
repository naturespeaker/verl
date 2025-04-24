# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Megatron-LM SFT Trainer with LoRA support.

This trainer is designed to work with Megatron-LM models and integrates
PEFT for LoRA fine-tuning. It draws inspiration from FSDPSFTTrainer
and OpenRLHF's SFTTrainer.

NOTE: This implementation contains placeholders for Megatron-LM specific
      API calls (e.g., initialization, model loading, forward/backward,
      checkpointing). These need to be filled in based on the actual
      Megatron-LM environment and the specific model implementation used
      (e.g., verl.models.llama.megatron).
"""

import os
import logging
from contextlib import nullcontext # Or Megatron-specific context managers

import torch
import torch.distributed as dist
# === Megatron-LM Imports (!!! USER ACTION REQUIRED: Adjust these imports based on your Megatron library structure !!!) ===
# Assuming these are standard Megatron-Core or similar imports - VERIFY PATHS
from megatron.core.models.gpt.gpt_model import GPTModel # Example, adjust to your model class (e.g., Llama specific) # PYLINT_IGNORE: import-error, no-name-in-module
from megatron.training import get_args, print_rank_0, get_tokenizer, get_model # PYLINT_IGNORE: import-error, no-name-in-module
from megatron.training.initialize import initialize_megatron # PYLINT_IGNORE: import-error, no-name-in-module
from megatron.training.arguments import core_transformer_config_from_args # PYLINT_IGNORE: import-error, no-name-in-module
from megatron.core.transformer.transformer_config import TransformerConfig # PYLINT_IGNORE: import-error, no-name-in-module
# Corrected DDP import for Megatron-Core
from megatron.core.distributed import DistributedDataParallel as DDP # PYLINT_IGNORE: import-error, no-name-in-module
# Attempting core import for optimizer helper - VERIFY THIS PATH
from megatron.core.optimizer import get_megatron_optimizer # PYLINT_IGNORE: import-error, no-name-in-module
# from megatron.core.optimizer import OptimizerConfig # If using core config # PYLINT_IGNORE: import-error, no-name-in-module
# Attempting core import for scheduler - VERIFY THIS PATH
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler # PYLINT_IGNORE: import-error, no-name-in-module
from megatron.training.checkpointing import load_checkpoint, save_checkpoint # PYLINT_IGNORE: import-error, no-name-in-module # Assuming checkpointing remains in training
from megatron.training.utils import average_losses_across_data_parallel_group, get_ltor_masks_and_position_ids # PYLINT_IGNORE: import-error, no-name-in-module # Assuming utils remain in training
from megatron.training.training import train_step, setup_model_and_optimizer # Useful helper # PYLINT_IGNORE: import-error, no-name-in-module
# Removed biencoder import as requested
# === End Megatron-LM Imports ===
from torch import nn, optim # Consider using Megatron's optimizer if needed
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from peft import LoraConfig, TaskType, get_peft_model, PeftModel # Added for LoRA and type checking
# Use Megatron's tokenizer or HF's AutoTokenizer if compatible
# from transformers import AutoTokenizer

# verl imports
import verl.utils.hdfs_io as hdfs_io
from verl.utils.dataset import SFTDataset # Or MultiTurnSFTDataset, or custom
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset # Example
from verl.utils.fs import copy_to_local # If needed for tokenizer/model paths
from verl.utils.torch_functional import get_cosine_schedule_with_warmup # Or Megatron's scheduler
from verl.utils.tracking import Tracking
# !!! USER ACTION REQUIRED: Remove the import below and define/import the actual model_provider function !!!
# from verl.utils.megatron_utils import get_model_provider # Assuming a helper exists # PYLINT_IGNORE: import-error, no-name-in-module
from omegaconf import OmegaConf, DictConfig, ListConfig # For config handling and type checking

# Added helper from FSDP trainer for Hydra config conversion
def convert_to_regular_types(obj):
    """Convert Hydra configs and other special types to regular Python types."""
    if isinstance(obj, (ListConfig, DictConfig)):
        return {k: convert_to_regular_types(v) for k, v in obj.items()} if isinstance(obj, DictConfig) else list(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_regular_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_regular_types(v) for k, v in obj.items()}
    return obj

logger = logging.getLogger(__name__)
# Set logging level based on environment or config
logger.setLevel(os.getenv("VERL_MEGATRON_SFT_LOGGING_LEVEL", "INFO"))

# Use Megatron's print_rank_0 directly if imported
# def print_rank_0(*args, **kwargs):
#     if not dist.is_initialized() or dist.get_rank() == 0:
#         print(*args, **kwargs)

class MegatronSFTTrainer:
    """
    Trainer for Supervised Fine-Tuning (SFT) using Megatron-LM with LoRA support.
    """
    def __init__(self, config):
        """
        Initializes the MegatronSFTTrainer.

        Args:
            config: Configuration object (e.g., Hydra OmegaConf) containing
                    trainer, model, data, optim settings.
        """
        self.config = config # Hydra config
        # --- Megatron Initialization ---
        # Initialize Megatron. This sets up distributed groups, timers, etc.
        # Pass relevant config values as defaults or overrides.
        # Need to map Hydra config to Megatron args format if necessary.
        # Example mapping (adjust based on your config structure):
        megatron_args_defaults = {
            'tokenizer_type': 'SentencePieceTokenizer', # Or AutoTokenizer
            'tensor_model_parallel_size': config.model.tensor_model_parallel_size,
            'pipeline_model_parallel_size': config.model.pipeline_model_parallel_size,
            'num_layers': config.model.num_layers,
            'hidden_size': config.model.hidden_size,
            'num_attention_heads': config.model.num_attention_heads,
            'seq_length': config.data.seq_length,
            'max_position_embeddings': config.data.seq_length,
            'micro_batch_size': config.data.micro_batch_size_per_gpu,
            'global_batch_size': config.data.train_batch_size * self._get_dp_world_size(config), # Calculate global bsz
            'rampup_batch_size': None, # Or from config
            'lr': config.optim.lr,
            'train_iters': None, # Will be calculated later
            'lr_decay_iters': None, # Will be calculated later
            'lr_decay_style': config.optim.get("lr_scheduler", "cosine"),
            'weight_decay': config.optim.weight_decay,
            'adam_beta1': config.optim.betas[0],
            'adam_beta2': config.optim.betas[1],
            'adam_eps': config.optim.get("eps", 1e-8),
            'clip_grad': config.optim.clip_grad,
            'bf16': config.model.get("dtype", "bfloat16") == "bfloat16",
            'fp16': config.model.get("dtype", "bfloat16") == "float16",
            'load': config.model.partial_pretrain, # Checkpoint loading path
            'save': config.trainer.get("default_local_dir", "./checkpoints"), # Checkpoint saving path
            'save_interval': config.trainer.save_steps, # Checkpoint save interval (in steps)
            'log_interval': config.trainer.logging_steps,
            'eval_interval': config.trainer.eval_steps,
            'eval_iters': config.trainer.get("eval_iters", 10), # Number of validation batches
            'gradient_accumulation_fusion': config.model.get("gradient_accumulation_fusion", False),
            'use_flash_attn': config.model.get("use_flash_attention_2", False),
            # Add other necessary Megatron args from your config
        }
        # Remove None values to avoid overriding Megatron defaults unnecessarily
        megatron_args_defaults = {k: v for k, v in megatron_args_defaults.items() if v is not None}

        initialize_megatron(args_defaults=megatron_args_defaults)

        self.args = get_args() # Get the final Megatron args after initialization
        # --- End Megatron Initialization ---

        # Get distributed info from Megatron args
        self.data_parallel_rank = self.args.rank # Megatron's global rank might be used for DP rank in some setups
        self.data_parallel_group = self.args.data_parallel_group
        self.data_parallel_world_size = self.args.data_parallel_world_size
        print_rank_0(f"Megatron Initialized. DP World Size: {self.data_parallel_world_size}, Rank: {self.data_parallel_rank}")



        self._build_tokenizer()
        self._build_dataloader()
        self._build_model_optimizer_scheduler()
        self._build_tracking()

        print_rank_0(f"MegatronSFTTrainer initialized. Megatron Args:\n{self.args}")
        if self.data_parallel_rank == 0:
             print(f"Hydra Config:\n{OmegaConf.to_yaml(config)}")

    def _get_dp_world_size(self, config):
        """Helper to estimate DP world size before Megatron init."""
        # This is a rough estimate, Megatron's init is the source of truth
        # Assumes standard DP/TP/PP setup
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        tp = config.model.tensor_model_parallel_size
        pp = config.model.pipeline_model_parallel_size
        dp = world_size // (tp * pp)
        return dp

    def _build_tokenizer(self):
        """Builds the tokenizer."""
        print_rank_0("Building tokenizer...")
        # Prefer tokenizer path from config if specified
        tokenizer_path = self.config.model.get("tokenizer_path", self.config.model.partial_pretrain)
        # copy_from_hdfs = self.config.model.get("copy_tokenizer_from_hdfs", False) # Example config
        # if copy_from_hdfs and tokenizer_path.startswith("hdfs://"):
        #     local_tokenizer_path = copy_to_local(src=tokenizer_path, verbose=(self.data_parallel_rank == 0))
        # else:
        #     local_tokenizer_path = tokenizer_path
        local_tokenizer_path = tokenizer_path # Assume local for now

        # --- Tokenizer Loading ---
        # Use Megatron's get_tokenizer() which should be initialized by initialize_megatron
        self.tokenizer = get_tokenizer()
        # --- End Tokenizer Loading ---

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print_rank_0(f"Set tokenizer pad_token to eos_token ({self.tokenizer.pad_token})")

        print_rank_0(f"Tokenizer built. Vocab size: {self.tokenizer.vocab_size}")


    def _build_dataloader(self):
        """Builds the training and validation dataloaders."""
        print_rank_0("Building dataloaders...")
        config = self.config.data

        # Determine dataset class (similar to FSDPSFTTrainer)
        if config.custom_cls.get("path", None):
             from verl.utils.import_utils import load_extern_type
             dataset_cls = load_extern_type(config.custom_cls.path, config.custom_cls.name)
        elif config.get("multiturn", {}).get("enable", False):
             dataset_cls = MultiTurnSFTDataset
        else:
             dataset_cls = SFTDataset
        print_rank_0(f"Using dataset class: {dataset_cls.__name__}")

        # Create datasets
        # Ensure dataset config is passed correctly
        dataset_kwargs = {"tokenizer": self.tokenizer, "config": config}
        self.train_dataset = dataset_cls(
            parquet_files=config.train_files, **dataset_kwargs
        )
        self.val_dataset = dataset_cls(
            parquet_files=config.val_files, **dataset_kwargs
        )
        print_rank_0(f"Train dataset size: {len(self.train_dataset)}, Val dataset size: {len(self.val_dataset)}")

        # Create samplers using Megatron's data parallel group info
        # Note: Megatron's data loaders might handle sampling internally,
        # check if explicit samplers are needed or if they conflict.
        # Using standard DistributedSampler with Megatron's DP group for now.
        self.train_sampler = DistributedSampler(
            self.train_dataset,
            shuffle=True,
            num_replicas=self.args.data_parallel_world_size,
            rank=self.args.data_parallel_rank,
            drop_last=True
        )
        self.val_sampler = DistributedSampler(
            self.val_dataset,
            shuffle=False,
            num_replicas=self.args.data_parallel_world_size,
            rank=self.args.data_parallel_rank,
            drop_last=False
        )

        # Normalize batch sizes by DP world size
        # Ensure global batch size is divisible by DP size
        global_train_bsz = config.train_batch_size
        global_eval_bsz = config.eval_batch_size
        dp_size = self.args.data_parallel_world_size

        assert global_train_bsz % dp_size == 0, \
            f"Global train batch size {global_train_bsz} must be divisible by DP size {dp_size}"
        assert global_eval_bsz % dp_size == 0, \
            f"Global eval batch size {global_eval_bsz} must be divisible by DP size {dp_size}"

        local_train_bsz = global_train_bsz // dp_size
        local_eval_bsz = global_eval_bsz // dp_size
        print_rank_0(f"Global Train BSZ: {global_train_bsz}, Local Train BSZ: {local_train_bsz}")
        print_rank_0(f"Global Eval BSZ: {global_eval_bsz}, Local Eval BSZ: {local_eval_bsz}")


        # Create dataloaders
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=local_train_bsz, # Use local batch size
            sampler=self.train_sampler,
            num_workers=config.get("num_workers", 8),
            pin_memory=True,
            drop_last=True,
        )
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=local_eval_bsz, # Use local batch size
            sampler=self.val_sampler,
            num_workers=config.get("num_workers", 8),
            pin_memory=True,
            drop_last=False,
        )
        print_rank_0("Dataloaders built.")

    def _build_model_optimizer_scheduler(self):
        """Builds the Megatron model, optimizer, and learning rate scheduler."""
        print_rank_0("Building model, optimizer, and scheduler...")
        model_config = self.config.model
        optim_config = self.config.optim

        # --- Megatron Model Loading ---
        # Use Megatron's get_model, which typically takes a model_provider function.
        # The model_provider should return an instance of the Megatron model (e.g., GPTModel).
        # We assume a helper `get_model_provider` exists in `verl.utils.megatron_utils`
        # that handles config mapping and returns the correct provider function.
        # !!! USER ACTION REQUIRED: Define or import the correct model_provider function here !!!
        # Example placeholder function:
        def model_provider(pre_process=True, post_process=True):
             """
             Build the model architecture (placeholder).

             !!! USER ACTION REQUIRED !!!
             This function MUST be implemented to return an instance of the correct
             Megatron model class based on your configuration (e.g., Llama, DeepSeek).
             It should use `self.args` (derived from the Hydra config during init)
             to determine model parameters and architecture.
             The actual checkpoint weights specified in `config.model.partial_pretrain`
             (mapped to `self.args.load`) will be loaded into this architecture by
             Megatron's `get_model` function later.
             """
             print_rank_0('Building Megatron model architecture (using placeholder provider)...')
             megatron_transformer_config: TransformerConfig = core_transformer_config_from_args(self.args) # PYLINT_IGNORE: name-error (Import added)

             # Example logic (replace with your actual implementation):
             # !!! NOTE: Pylint errors regarding missing arguments for GPTModel constructor
             # are expected here as this is a placeholder. USER ACTION REQUIRED: Implement the
             # correct model instantiation based on 'model_type' below.
             model_type = self.config.model.get("type", "llama") # Get model type from Hydra config
             print_rank_0(f"Attempting to build model of type: {model_type}")
 
             # !!! USER ACTION REQUIRED: Replace the following placeholder logic with your actual model loading !!!
             if model_type == "llama":
                 # Example: Import and instantiate your Llama model class
                 # from verl.models.llama.megatron import LlamaModel # Adjust import path
                 # model = LlamaModel(config=megatron_transformer_config, ...)
                 raise NotImplementedError("Placeholder: Llama model instantiation not implemented in model_provider.")
             elif model_type == "deepseek":
                 # Example: Import and instantiate your DeepSeek model class
                 # from verl.models.deepseek.megatron import DeepSeekModel # Adjust import path
                 # model = DeepSeekModel(config=megatron_transformer_config, ...)
                 raise NotImplementedError("Placeholder: DeepSeek model instantiation not implemented in model_provider.")
             elif model_type == "gpt": # Fallback to generic GPT for example
                 print_rank_0("Warning: Falling back to generic GPTModel placeholder in model_provider.")
                 model = GPTModel(
                     config=megatron_transformer_config,
                     parallel_output=True,
                     pre_process=pre_process,
                     post_process=post_process
                 )
             else:
                 raise ValueError(f"Unsupported model type '{model_type}' specified in config.")

             return model
        # --- End model_provider Definition ---

        # Get the Megatron transformer config
        megatron_transformer_config = core_transformer_config_from_args(self.args)

        # `get_model` handles model instantiation and potentially loading checkpoints
        # if `self.args.load` is set. It returns a list of models for pipeline parallelism.
        # For non-pipeline models, it's usually a list with one element.
        # Removed incorrect 'config' argument from get_model call
        model_list = get_model(model_provider)
        if len(model_list) == 1:
             self.model = model_list[0]
        else:
             # Handle pipeline parallelism if necessary (requires more complex setup)
             raise NotImplementedError("Pipeline Parallelism model handling not fully implemented in this trainer.")
             # self.model = model_list # Keep as list for pipeline stages

        print_rank_0(f"Megatron model ({type(self.model).__name__}) built.")
        # Note: Checkpoint loading is typically handled inside `get_model` or `setup_model_and_optimizer`
        # based on `self.args.load`. We might not need the explicit load call here.
        # --- End Megatron Model Loading ---


        # Apply LoRA if configured
        if model_config.get("lora_rank", 0) > 0:
            print_rank_0(f"Applying LoRA with rank {model_config.lora_rank}")
            # Determine target modules for Megatron Llama (needs inspection of the model structure)
            # Example: target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"] # Common Megatron names
            # !!! USER ACTION REQUIRED: Verify these target modules match YOUR Megatron Llama implementation !!!
            # Inspect your model architecture (e.g., print(self.model) before PEFT wrapping) to find the correct names.
            # Common HF Llama targets: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            # Megatron might use different names like "query_key_value", "dense", "attention.dense", "mlp.dense_h_to_4h", etc.
            # Using common Megatron names as default, VERIFY THESE:
            default_targets = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
            target_modules_config = model_config.get("target_modules", default_targets)
            # Convert Hydra ListConfig to regular list if necessary
            target_modules = convert_to_regular_types(target_modules_config)
            print_rank_0(f"Using LoRA target modules (VERIFY THESE): {target_modules}")

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, # Assuming Causal LM task
                r=model_config.lora_rank,
                lora_alpha=model_config.lora_alpha,
                target_modules=target_modules, # Use converted list
                lora_dropout=model_config.get("lora_dropout", 0.0),
                bias="none",
                # modules_to_save: Add if training embeddings/lm_head with LoRA
                # modules_to_save=["embed_tokens", "lm_head"] if model_config.get("lora_train_embedding") else None,
            )

            # Ensure input grads are enabled for LoRA
            self.model.enable_input_require_grads() # Uncommented

            self.model = get_peft_model(self.model, lora_config)
            print_rank_0("PEFT model created.")
            self.model.print_trainable_parameters()

        # --- Megatron Model Parallel Wrapping ---
        # Wrap the model (potentially PEFT-modified) with Megatron's DDP.
        # `setup_model_and_optimizer` might handle this, or we do it manually.
        # Manual wrapping:
        self.model = DDP(
            config=megatron_transformer_config, # Pass Megatron config
            model=self.model,
            # data_parallel_group=self.args.data_parallel_group, # Usually inferred
            # gradient_accumulation_fusion=self.args.gradient_accumulation_fusion # Usually inferred
        )
        print_rank_0("Wrapped model with Megatron DDP.")
        # --- End Megatron Model Parallel Wrapping ---

        # Enable gradient checkpointing
        # PEFT handles GC wrapping if LoRA is used and GC is enabled in the base model config or PEFT config.
        # If not using LoRA, enable Megatron's or HF's native GC here.
        if model_config.enable_gradient_checkpointing:
            # Check if PEFT already enabled it
            already_enabled = getattr(self.model.config, "gradient_checkpointing", False) or \
                              (hasattr(self.model, "is_gradient_checkpointing") and self.model.is_gradient_checkpointing)

            if not already_enabled:
                 # Enable it on the base model (or PEFT model if it supports it directly)
                 self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}) # HF style
                 print_rank_0("Gradient checkpointing enabled (explicitly).")
            else:
                 print_rank_0("Gradient checkpointing already enabled (likely by PEFT or base config).")


        # Build Optimizer
        print_rank_0("Building optimizer...")
        # --- Megatron Optimizer ---
        # Use Megatron's optimizer function. It handles parameter grouping for weight decay.
        # It expects the model *before* DDP wrapping if creating groups manually,
        # but `get_megatron_optimizer` might handle the DDP-wrapped model directly.
        # Check Megatron's documentation for the specific version.
        # !!! USER ACTION REQUIRED: Verify if `get_megatron_optimizer` correctly handles PEFT models !!!
        # It should ideally only consider parameters where `param.requires_grad is True`.
        # If it doesn't automatically handle this for PEFT models, you might need to filter parameters manually:
        # trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        # self.optimizer = get_megatron_optimizer(trainable_params) # Or adapt get_megatron_optimizer
        # Assuming `get_megatron_optimizer` works correctly with the DDP-wrapped PEFT model for now:
        print_rank_0("Attempting to build optimizer using get_megatron_optimizer. VERIFY that it correctly handles PEFT parameters.")
        self.optimizer = get_megatron_optimizer(self.model)
        # --- End Megatron Optimizer ---
        print_rank_0(f"Optimizer built: {type(self.optimizer).__name__}")


        # Build Learning Rate Scheduler
        print_rank_0("Building LR scheduler...")
        # Calculate steps considering gradient accumulation if Megatron uses it
        grad_accum_steps = self.args.gradient_accumulation_steps
        self.steps_per_epoch = len(self.train_dataloader) // grad_accum_steps
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs
        num_warmup_steps = int(self.total_steps * optim_config.warmup_steps_ratio)
        print_rank_0(f"Grad accum steps: {grad_accum_steps}, Steps per epoch: {self.steps_per_epoch}")
        print_rank_0(f"Total steps: {self.total_steps}, Warmup steps: {num_warmup_steps}")

        # --- Megatron LR Scheduler ---
        # Megatron's scheduler is often integrated with the optimizer or obtained separately.
        # `OptimizerParamScheduler` is common.
        self.lr_scheduler = OptimizerParamScheduler(
             optimizer=self.optimizer,
             max_lr=self.args.lr,
             min_lr=self.args.min_lr,
             lr_warmup_steps=self.args.lr_warmup_iters, # Use iters if available
             lr_decay_steps=self.args.lr_decay_iters,   # Use iters if available
             lr_decay_style=self.args.lr_decay_style,
             start_wd=self.args.start_weight_decay,
             end_wd=self.args.end_weight_decay,
             wd_incr_steps=self.args.wd_incr_steps,
             wd_incr_style=self.args.wd_incr_style,
             use_checkpoint_opt_param_scheduler=self.args.use_checkpoint_opt_param_scheduler,
             override_opt_param_scheduler=self.args.override_opt_param_scheduler
        )
        # --- End Megatron LR Scheduler ---
        print_rank_0(f"LR Scheduler built: {self.args.lr_decay_style}")

        # --- Setup Model and Optimizer (Optional Helper) ---
        # Megatron's `setup_model_and_optimizer` can handle model DDP wrapping,
        # optimizer creation, and loading optimizer/scheduler states from checkpoint.
        # If used, it might replace some of the manual steps above.
        # Example:
        # self.model, self.optimizer, self.lr_scheduler = setup_model_and_optimizer(
        #     model_provider=model_provider, # Provider function
        #     config=megatron_transformer_config,
        #     # Pass other args like model_type, teacher_model, etc. if needed
        # )
        # print_rank_0("Model, Optimizer, and Scheduler setup via Megatron helper.")
        # --- End Setup Model and Optimizer ---

        # Load checkpoint for optimizer and scheduler if resuming
        # This might be handled by `setup_model_and_optimizer` or needs manual call
        if self.args.load is not None:
            # `load_checkpoint` loads model, optimizer, and scheduler states
            # It's often called implicitly by `get_model` or `setup_model_and_optimizer`
            # If manual loading is needed:
            # If manual loading is needed:
            # NOTE: Loading LoRA adapters requires separate handling. Megatron's `load_checkpoint`
            # loads the base model and potentially optimizer/scheduler state.
            # LoRA adapters need to be loaded *after* the base model is loaded and *before*
            # the optimizer is potentially re-initialized based on the loaded state,
            # potentially using `PeftModel.from_pretrained(self.model, lora_checkpoint_path)`.
            # This requires careful coordination of checkpoint paths and loading steps.
            iteration = load_checkpoint(self.model, self.optimizer, self.lr_scheduler)
            if iteration > 0:
                print_rank_0(f"Loaded Base Model/Optimizer/Scheduler state from Megatron checkpoint at iteration {iteration}")
                # !!! USER ACTION REQUIRED: Implement LoRA adapter loading !!!
                # If resuming LoRA training, load the adapter weights *after* the base model is loaded.
                # Example:
                # if model_config.get("lora_rank", 0) > 0:
                #     lora_checkpoint_path = os.path.join(self.args.load, f"iter_{iteration:07d}", "lora_adapter")
                #     if os.path.exists(lora_checkpoint_path):
                #         print_rank_0(f"Attempting to load LoRA adapter from: {lora_checkpoint_path}")
                #         # Ensure the model is already a PeftModel before loading adapters
                #         if isinstance(self.model, PeftModel) or (hasattr(self.model, 'module') and isinstance(self.model.module, PeftModel)):
                #             # Access the underlying PeftModel instance correctly (might need adjustment based on DDP wrapper)
                #             peft_model_to_load = self.model.module if hasattr(self.model, 'module') else self.model
                #             peft_model_to_load.load_adapter(lora_checkpoint_path, adapter_name="default") # Or the name used during saving
                #             print_rank_0("LoRA adapter loaded successfully.")
                #         else:
                #             print_rank_0("ERROR: Model is not a PeftModel instance, cannot load LoRA adapter.")
                #     else:
                #         print_rank_0(f"WARNING: LoRA checkpoint path not found: {lora_checkpoint_path}")
            else: # Corresponds to `if iteration > 0:`
                print_rank_0("No Megatron checkpoint state found or starting from iteration 0. Starting training from scratch (or base weights).")
        else:
            # Indent iteration assignment under this else
            iteration = 0 # Start from beginning
        # Ensure this assignment is aligned with the if/else block starting at L485
        self.start_step = iteration * self.args.gradient_accumulation_steps # Adjust starting step based on loaded iteration

    # Ensure this method definition is aligned with other methods like `_build_model_optimizer_scheduler`
    def _build_tracking(self):
        # Indent content of the method
        if self.data_parallel_rank == 0 and self.config.trainer.get("logger", None):
            logger_backend = self.config.trainer.logger
            print_rank_0(f"Initializing tracking: {logger_backend}")
            # Convert OmegaConf to dict for logging if necessary
            # Ensure OmegaConf is imported at the top
            try:
                # Use OmegaConf if it's the expected type
                if isinstance(self.config, OmegaConf):
                     log_config = OmegaConf.to_container(self.config, resolve=True)
                else: # Otherwise, assume it's already a dict-like object
                     log_config = dict(self.config)
            except Exception as e: # Catch specific exceptions if possible
                print_rank_0(f"Warning: Could not convert config to dict for logging: {e}")
                log_config = {} # Fallback to empty dict

            self.tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=logger_backend,
                config=log_config # Log the config
            )
        # Ensure this else is aligned with the if at L512
        else:
            # Indent the assignment under this else
            self.tracking = None

    def _compute_loss(self, logits, labels, loss_mask):
        """
        Computes the Cross Entropy loss, masking out padding tokens.

        Args:
            logits: Model output logits (batch_size, seq_len, vocab_size).
            labels: Ground truth labels (batch_size, seq_len).
            loss_mask: Mask indicating which tokens to include in loss (batch_size, seq_len).

        Returns:
            Average loss over valid tokens in the batch.
        """
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        # Shift logits and labels for next token prediction
        # Logits: (B, S, V) -> (B, S-1, V)
        # Labels: (B, S) -> (B, S-1)
        # Mask:   (B, S) -> (B, S-1)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_loss_mask = loss_mask[..., 1:].contiguous().float() # Ensure float for multiplication

        # Flatten the tokens and compute loss: (B * (S-1), V) and (B * (S-1),)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Reshape and apply mask: (B, S-1)
        loss = loss.view(shift_labels.size()) * shift_loss_mask

        # Calculate average loss over the masked tokens
        # Avoid division by zero if mask is all False
        num_valid_tokens = shift_loss_mask.sum()
        average_loss = loss.sum() / (num_valid_tokens + 1e-8)
        return average_loss

    def training_step(self, batch):
        """Performs a single training step."""
        self.model.train()

        # Move batch to device (assuming CUDA)
        # Consider non_blocking=True for potential performance gain
        input_ids = batch["input_ids"].cuda(non_blocking=True)
        attention_mask = batch["attention_mask"].cuda(non_blocking=True)
        # SFT labels are typically the input_ids themselves
        labels = input_ids
        loss_mask = batch["loss_mask"].cuda(non_blocking=True) # Mask for padding/prompt tokens

        # --- Megatron Forward/Backward ---
        # Define the forward step function required by Megatron's train_step
        def forward_step(data_iterator, model):
            """Forward step function."""
            # Get data from iterator (assuming iterator yields batches like ours)
            # Megatron's train_step might handle the iterator differently.
            # This adaptation assumes we pass a single batch wrapped in an iterator.
            try:
                 batch = next(data_iterator)
            except StopIteration:
                 # This might happen if train_step calls forward_step multiple times
                 # for microbatches within a single global batch. Needs careful handling
                 # based on how train_step consumes the iterator.
                 # Returning None might signal the end. Check Megatron's implementation.
                 return None, None # Or raise an error

            # Move batch to device (redundant if already done, but safe)
            input_ids = batch["input_ids"].cuda(non_blocking=True)
            attention_mask = batch["attention_mask"].cuda(non_blocking=True)
            labels = input_ids # SFT labels
            loss_mask = batch["loss_mask"].cuda(non_blocking=True)

            # Get position_ids and attention_mask in the format Megatron expects
            # This might depend on the specific model implementation (e.g., Llama)
            # attn_mask_type = self.args.attn_mask_type # This argument is removed based on user feedback for megatron-core 0.13.0rc0
            attention_mask_meg, loss_mask_meg, position_ids = get_ltor_masks_and_position_ids(
                data=input_ids,
                eod_token=self.tokenizer.eos_token_id, # Use EOS as EOD
                reset_position_ids=self.args.reset_position_ids,
                reset_attention_mask=self.args.reset_attention_mask,
                eod_mask_loss=self.args.eod_mask_loss
                # attn_mask_type argument removed
            )
            # Note: The loss_mask returned by get_ltor_masks_and_position_ids might
            # differ from our dataset's loss_mask. We should likely use our dataset's mask.
            # Verify how eod_mask_loss interacts. Using our loss_mask for now.

            # Model forward pass (Megatron DDP model handles autocast internally)
            # The model might be a list in case of pipeline parallelism
            model_module = model[0] if isinstance(model, list) else model
            output_tensor = model_module(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask_meg # Use Megatron's mask format
                # labels=None # Usually computed separately from logits
            )

            # Loss computation
            # output_tensor might be logits or a tuple (logits, other_outputs)
            # Assume output_tensor contains logits for now
            logits = output_tensor.to(torch.float32) # Ensure float32 for loss calc
            loss = self._compute_loss(logits, labels, loss_mask) # Use our loss mask

            # Reduce loss across tensor/pipeline parallel groups if needed (usually handled by train_step)
            # Average loss across microbatches (handled by train_step)

            # Return loss tensor and a dictionary for logging (keys should match train_step expectations)
            # Example: {'lm_loss': loss}
            return loss, {'lm_loss': loss}

        # Wrap the current batch in an iterator for train_step
        batch_iterator = iter([batch])

        # Call Megatron's train_step
        # It handles forward, backward, optimizer step, gradient clipping, and LR scheduling.
        # It operates on microbatches internally based on gradient_accumulation_steps.
        # We pass the DDP-wrapped model.
        loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = train_step(
            forward_step_func=forward_step,
            data_iterator=batch_iterator,
            model=self.model, # Pass the DDP-wrapped model
            optimizer=self.optimizer,
            opt_param_scheduler=self.lr_scheduler # Pass Megatron's scheduler
        )
        # --- End Megatron Forward/Backward ---

        # Process results
        loss = loss_dict.get('lm_loss', torch.tensor(0.0)) # Get the relevant loss

        # Average loss across data parallel group (train_step might already do this, check)
        # If not, uncomment the line below:
        # loss = average_losses_across_data_parallel_group([loss.item()])[0]

        lr = self.optimizer.param_groups[0]['lr'] # Get current LR from optimizer
        metrics = {"train/loss": loss.item(), "train/lr": lr}
        if grad_norm is not None:
             metrics["train/grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        if skipped_iter:
             metrics["train/skipped_iters"] = 1 # Log skipped iteration if grad norm was inf/nan

        return metrics


    def validation_step(self, batch):
        """Performs a single validation step."""
        self.model.eval() # Set model to evaluation mode
        with torch.no_grad(): # Disable gradient calculations
            # Move batch to device
            input_ids = batch["input_ids"].cuda(non_blocking=True)
            attention_mask = batch["attention_mask"].cuda(non_blocking=True)
            labels = input_ids
            loss_mask = batch["loss_mask"].cuda(non_blocking=True)

            # Forward pass with autocast
            model_dtype = self.model.dtype
            with torch.autocast(device_type='cuda', dtype=model_dtype, enabled=(model_dtype != torch.float32)):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False
                )
                logits = outputs.logits
                # Compute loss in float32
                loss = self._compute_loss(logits.to(torch.float32), labels, loss_mask)

            # Synchronize loss across data parallel ranks
            reduced_loss = loss.detach().clone()
            dist.all_reduce(reduced_loss, op=dist.ReduceOp.AVG)

        return {"val/loss": reduced_loss.item()}

    def evaluate(self, global_step):
        """
        Runs evaluation on the validation dataset.

        Args:
            global_step (int): The current global training step for logging.
        """
        print_rank_0(f"Running validation at global step {global_step}...")
        self.model.eval()
        all_val_losses = []
        # Use try-finally to ensure model is set back to train mode
        try:
            val_iter = iter(self.val_dataloader)
            val_progress_bar = tqdm(
                range(len(self.val_dataloader)),
                disable=(self.args.rank != 0), # Use Megatron rank for disabling tqdm
                desc=f"Validation Step {global_step}"
            )

            for i in val_progress_bar:
                try:
                    batch = next(val_iter)
                    metrics = self.validation_step(batch)
                    all_val_losses.append(metrics["val/loss"])
                    val_progress_bar.set_postfix({"loss": metrics["val/loss"]})
                except StopIteration:
                    break # Should not happen with DataLoader length
        finally:
            self.model.train() # Set model back to training mode
        # Calculate average validation loss
        avg_val_loss = sum(all_val_losses) / len(all_val_losses) if all_val_losses else 0.0
        print_rank_0(f"Validation finished at global step {global_step}. Average Loss: {avg_val_loss:.4f}")
        metrics = {"val/loss": avg_val_loss}

        # Log validation metrics using the provided global_step
        if self.tracking:
            # Add global_step for proper logging association
            log_metrics = {f"validation/{k}": v for k, v in metrics.items()}
            log_metrics["global_step"] = global_step
            self.tracking.log(log_metrics)

        return metrics

    def fit(self):
        """Main training loop."""
        print_rank_0("Starting training...")
        global_step = 0
        # Start from the step loaded from checkpoint or 0
        global_step = self.start_step
        start_epoch = global_step // self.steps_per_epoch
        start_step_in_epoch = global_step % self.steps_per_epoch

        # Calculate total training steps based on Megatron's train_iters if available
        if self.args.train_iters:
             self.total_training_steps = self.args.train_iters * self.args.gradient_accumulation_steps
        elif self.config.trainer.get("total_training_steps") is not None:
             self.total_training_steps = self.config.trainer.total_training_steps
        else:
             self.total_training_steps = self.steps_per_epoch * self.config.trainer.total_epochs
        print_rank_0(f"Total training steps planned: {self.total_training_steps}")
        print_rank_0(f"Starting from Global Step: {global_step}, Epoch: {start_epoch+1}, Step in Epoch: {start_step_in_epoch}")

        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            print_rank_0(f"Starting Epoch {epoch+1}/{self.config.trainer.total_epochs}")
            self.train_sampler.set_epoch(epoch)

            # Adjust range if starting mid-epoch
            steps_in_epoch_range = range(len(self.train_dataloader))
            if epoch == start_epoch:
                 steps_in_epoch_range = range(start_step_in_epoch, len(self.train_dataloader))
                 # Need to advance the iterator to the correct starting point
                 train_iter = iter(self.train_dataloader)
                 print_rank_0(f"Advancing dataloader to step {start_step_in_epoch} in epoch {epoch+1}")
                 for _ in range(start_step_in_epoch):
                     try:
                         next(train_iter)
                     except StopIteration:
                         print_rank_0("Warning: Could not advance dataloader to starting step.")
                         break
            else:
                 train_iter = iter(self.train_dataloader)


            progress_bar = tqdm(
                steps_in_epoch_range,
                total=len(self.train_dataloader),
                initial=start_step_in_epoch if epoch == start_epoch else 0,
                disable=(self.data_parallel_rank != 0),
                desc=f"Epoch {epoch+1}"
            )

            for step_in_epoch in progress_bar: # step_in_epoch is 0-based index within the epoch
                # --- Megatron Train Step (Placeholder) ---
                # This might replace the manual batch fetching and training_step call
                # loss_dict = train_step(...)
                # train_metrics = {"train/loss": loss_dict['lm_loss'].item(), ...}
                # --- End Megatron Train Step ---

                # --- Standard PyTorch Train Step ---
                try:
                    # TODO: Handle gradient accumulation steps if needed
                    batch = next(train_iter)
                    train_metrics = self.training_step(batch)
                except StopIteration:
                    print_rank_0("Warning: Train dataloader iterator exhausted unexpectedly.")
                    break
                # --- End Standard PyTorch Train Step ---

                global_step += 1
                progress_bar.set_postfix(train_metrics)

                # Log metrics
                if self.tracking and global_step % self.config.trainer.logging_steps == 0:
                    self.tracking.log({**train_metrics, "global_step": global_step})

                # Validation (using Megatron's eval_interval)
                # Megatron's train_step might handle evaluation internally based on eval_interval.
                # If manual evaluation is desired based on trainer config:
                eval_steps = self.config.trainer.eval_steps
                if eval_steps > 0 and global_step % eval_steps == 0:
                    val_metrics = self.evaluate(global_step=global_step) # Pass global_step
                    # Logging is now handled inside evaluate()
                    # if self.tracking:
                    #     self.tracking.log({**val_metrics, "global_step": global_step})

                # Save checkpoint
                # Use >= 1 to ensure checkpoint happens at least once if save_steps is 1
                if global_step % self.config.trainer.save_steps == 0 and self.config.trainer.save_steps >= 1:
                    self.save_checkpoint(global_step)

                # Check for early stopping based on total steps
                if global_step >= self.total_training_steps:
                    print_rank_0(f"Reached total training steps ({self.total_training_steps}). Finishing training.")
                    break # Exit inner loop

            if global_step >= self.total_training_steps:
                break # Exit outer epoch loop as well

        print_rank_0("Training finished.")
        # Final checkpoint save
        if self.config.trainer.get("save_on_exit", True):
             self.save_checkpoint(global_step, final=True)

        if self.tracking:
             print_rank_0("Closing tracking...")
             try:
                 self.tracking.close() # Use close() instead of finish() # PYLINT_IGNORE: no-member
             except AttributeError:
                 print_rank_0("Warning: Tracking object does not have a close() method.")
             except Exception as e:
                 print_rank_0(f"Error closing tracking: {e}")


    def save_checkpoint(self, global_step, final=False):
        """Saves a checkpoint, handling LoRA adapters if enabled."""
        # Megatron uses 'iteration' which corresponds to optimizer steps.
        iteration = global_step // self.args.gradient_accumulation_steps
        megatron_save_dir = self.args.save # Base directory from Megatron args
        megatron_iter_dir = os.path.join(megatron_save_dir, f"iter_{iteration:07d}") # Specific iteration dir

        print_rank_0(f"Saving checkpoint for iteration {iteration} (global step {global_step}) to {megatron_iter_dir}...")

        # --- Megatron Checkpoint Saving ---
        # Saves base model (potentially frozen), optimizer, scheduler, args.
        save_checkpoint(iteration=iteration,
                        model=self.model, # Pass DDP-wrapped model
                        optimizer=self.optimizer,
                        opt_param_scheduler=self.lr_scheduler)
        print_rank_0(f"Megatron state saved for iteration {iteration}.")
        # --- End Megatron Checkpoint Saving ---

        # --- Save LoRA Adapters Separately (if LoRA enabled) ---
        lora_enabled = self.config.model.get("lora_rank", 0) > 0
        peft_model_instance = None

        # Try to access the underlying PEFT model instance.
        # This needs verification based on the specific Megatron DDP implementation.
        if hasattr(self.model, 'module'): # Standard PyTorch DDP/FSDP attribute
            if isinstance(self.model.module, PeftModel):
                peft_model_instance = self.model.module
        elif isinstance(self.model, PeftModel): # Check if the model itself is PEFT (e.g., if DDP wrapping failed or is different)
             peft_model_instance = self.model
        # Add other potential access methods if Megatron DDP uses a different attribute

        if lora_enabled and peft_model_instance:
            lora_save_path = os.path.join(megatron_iter_dir, "lora_adapter") # Save inside iter dir
            if self.args.rank == 0: # Save only on global rank 0
                print_rank_0(f"Saving LoRA adapter to {lora_save_path}...")
                os.makedirs(lora_save_path, exist_ok=True)
                try:
                    peft_model_instance.save_pretrained(lora_save_path)
                    # Save the tokenizer as well for convenience when loading adapter standalone
                    self.tokenizer.save_pretrained(lora_save_path)
                    print_rank_0("LoRA adapter and tokenizer saved successfully.")
                except Exception as e:
                    print_rank_0(f"ERROR: Failed to save LoRA adapter checkpoint to {lora_save_path}: {e}")
        elif lora_enabled:
             if self.args.rank == 0:
                 print_rank_0("WARNING: LoRA is enabled, but could not access the underlying PeftModel instance to save adapters separately. Check DDP wrapping.")
        # --- End Save LoRA Adapters ---

        # Barrier after all saving operations on rank 0 are initiated
        dist.barrier()

        # --- HDFS Upload (Optional) ---
        hdfs_dir = self.config.trainer.get("default_hdfs_dir", None)
        if self.args.rank == 0 and hdfs_dir:
             # Source is the Megatron checkpoint dir for this iteration (now contains adapters if saved)
             local_source_dir = megatron_iter_dir
             # Target path on HDFS (using iteration number for consistency)
             hdfs_target_path = os.path.join(hdfs_dir, f"iter_{iteration:07d}")
             print_rank_0(f"Attempting to copy checkpoint from {local_source_dir} to HDFS: {hdfs_target_path}")
             try:
                 # Ensure parent HDFS directory exists
                 parent_hdfs_dir = os.path.dirname(hdfs_target_path)
                 if parent_hdfs_dir != hdfs_dir and parent_hdfs_dir != '': # Handle base dir case
                      hdfs_io.makedirs(parent_hdfs_dir, exist_ok=True)
                 # Copy the entire directory content
                 hdfs_io.copy(src=local_source_dir, dst=hdfs_target_path, dirs_exist_ok=True)
                 print_rank_0("Checkpoint directory copied to HDFS.")
             except Exception as e:
                 print_rank_0(f"ERROR: Failed to copy checkpoint directory to HDFS: {e}")
             # No barrier needed here as rank 0 does the copy, others wait at the previous barrier.
        # --- End HDFS Upload ---

        print_rank_0(f"Checkpoint saving process complete for iteration {iteration}.")


# Example main entry point (adapt as needed, e.g., using Hydra)
# import hydra
# from omegaconf import DictConfig, OmegaConf

# @hydra.main(config_path="config", config_name="megatron_sft_config", version_base=None)
# def main(cfg: DictConfig):
#     print_rank_0("Starting Megatron SFT Training...")
#     print_rank_0(OmegaConf.to_yaml(cfg))

#     # --- Megatron Initialization Call (Placeholder) ---
#     # This needs to happen *before* the Trainer is initialized if Megatron handles
#     # distributed setup globally.
#     # initialize_megatron(args_defaults=...) # Pass relevant args from cfg
#     # --- End Megatron Initialization Call ---

#     # Ensure CUDA device is set after potential Megatron init
#     if dist.is_initialized():
#          torch.cuda.set_device(dist.get_rank())

#     # Create and run the trainer
#     trainer = MegatronSFTTrainer(cfg)
#     trainer.fit()
#     print_rank_0("Training finished successfully.")

# if __name__ == "__main__":
#     # main() # Uncomment to run with Hydra
#     pass # Add main execution logic here