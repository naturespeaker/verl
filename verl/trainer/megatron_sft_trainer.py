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
# === Megatron-LM Imports (PLACEHOLDERS - Adjust based on your Megatron setup) ===
# from megatron import get_args, print_rank_0, get_tokenizer
# from megatron.initialize import initialize_megatron
# from megatron.model import ModelType, DistributedDataParallel as DDP # Example imports
# from megatron.training import get_optimizer_param_groups, get_optimizer, get_learning_rate_scheduler, train_step # Example imports
# from megatron.checkpointing import load_checkpoint, save_checkpoint # Example imports
# from megatron.utils import average_losses_across_data_parallel_group, get_ltor_masks_and_position_ids # Example imports
# from megatron.arguments import core_transformer_config_from_args # Example import
# === End Megatron-LM Imports ===

from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from torch import nn, optim # Consider using Megatron's optimizer if needed
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
# Use Megatron's tokenizer or HF's AutoTokenizer if compatible
# from transformers import AutoTokenizer

# verl imports
import verl.utils.hdfs_io as hdfs_io
from verl.utils.dataset import SFTDataset # Or MultiTurnSFTDataset, or custom
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset # Example
from verl.utils.fs import copy_to_local # If needed for tokenizer/model paths
from verl.utils.torch_functional import get_cosine_schedule_with_warmup # Or Megatron's scheduler
from verl.utils.tracking import Tracking
# from verl.utils.megatron_utils import get_megatron_model_provider # Example helper

logger = logging.getLogger(__name__)
# Set logging level based on environment or config
logger.setLevel(os.getenv("VERL_MEGATRON_SFT_LOGGING_LEVEL", "INFO"))

# Placeholder for print_rank_0 if not using Megatron's
def print_rank_0(*args, **kwargs):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)

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
        self.config = config
        # self.args = get_args() # Get Megatron arguments if needed

        # --- Megatron Initialization (Placeholder) ---
        # This section needs to be adapted based on how Megatron is initialized
        # in your environment. It might involve calling initialize_megatron or similar.
        # Example:
        # initialize_megatron(args_defaults={'tokenizer_type': 'SentencePieceTokenizer', ...})
        # self.model_parallel_size = self.args.tensor_model_parallel_size
        # self.data_parallel_rank = self.args.rank # Megatron's global rank
        # self.data_parallel_group = self.args.data_parallel_group # Megatron's DP group
        # self.data_parallel_world_size = dist.get_world_size(group=self.data_parallel_group)
        # --- End Megatron Initialization ---

        # Use standard PyTorch distributed info for now (replace with Megatron specifics)
        # Ensure distributed environment is initialized
        if not dist.is_initialized():
             # Basic initialization, replace with Megatron's if it handles it
             dist.init_process_group(backend='nccl')
             torch.cuda.set_device(dist.get_rank()) # Crucial for Megatron/PyTorch

        self.data_parallel_rank = dist.get_rank()
        self.data_parallel_world_size = dist.get_world_size()
        print_rank_0(f"Initialized with DP rank {self.data_parallel_rank}/{self.data_parallel_world_size}")


        self._build_tokenizer()
        self._build_dataloader()
        self._build_model_optimizer_scheduler()
        self._build_tracking()

        print_rank_0(f"MegatronSFTTrainer initialized. Config:\n{config}")

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

        # --- Tokenizer Loading (Placeholder for Megatron Tokenizer) ---
        # If using Megatron's tokenizer:
        # self.tokenizer = get_tokenizer()
        # Or use HuggingFace tokenizer if compatible:
        from verl.utils import hf_tokenizer # Assuming this helper exists and works
        self.tokenizer = hf_tokenizer(
            local_tokenizer_path,
            trust_remote_code=self.config.model.trust_remote_code
        )
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

        # Create samplers (using Megatron's DP info if available)
        self.train_sampler = DistributedSampler(
            self.train_dataset,
            shuffle=True,
            num_replicas=self.data_parallel_world_size,
            rank=self.data_parallel_rank,
            drop_last=True
        )
        self.val_sampler = DistributedSampler(
            self.val_dataset,
            shuffle=False,
            num_replicas=self.data_parallel_world_size,
            rank=self.data_parallel_rank,
            drop_last=False # Usually don't drop last for validation
        )

        # Normalize batch sizes by DP world size
        # Ensure global batch size is divisible by DP size
        global_train_bsz = config.train_batch_size
        global_eval_bsz = config.eval_batch_size
        dp_size = self.data_parallel_world_size

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

        # --- Megatron Model Loading (Placeholder) ---
        # This is the most critical part to adapt.
        # It should use Megatron's functions to build or load the model,
        # considering model parallelism and potentially pipeline parallelism.
        # Example using a hypothetical provider function:
        # model_provider = get_megatron_model_provider(model_config) # Assumes this exists
        # megatron_transformer_config = core_transformer_config_from_args(self.args) # Get Megatron config
        # self.model = model_provider(
        #     model_type=ModelType.encoder_or_decoder, # Or appropriate type
        #     config=megatron_transformer_config,
        #     pre_process=True, # Megatron flags
        #     post_process=True
        # )
        # # Load checkpoint if specified
        # if model_config.partial_pretrain:
        #     iteration = load_checkpoint(self.model, None, None, load_arg='load', strict=False) # Megatron load
        #     print_rank_0(f"Loaded checkpoint from {model_config.partial_pretrain} at iteration {iteration}")
        # --- End Megatron Model Loading ---

        # --- Temporary Placeholder using HF Transformers (REMOVE THIS) ---
        # This is NOT correct for Megatron but keeps the structure.
        from transformers import AutoModelForCausalLM, AutoConfig
        local_model_path = model_config.partial_pretrain
        print_rank_0(f"Loading HF model placeholder from: {local_model_path}")
        hf_config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=model_config.trust_remote_code)
        # Set gradient checkpointing based on config BEFORE loading model if using HF
        hf_config.use_cache = False # Important for training
        # Gradient checkpointing is handled later by PEFT or Megatron's wrapper
        # if model_config.enable_gradient_checkpointing:
        #      hf_config.gradient_checkpointing = True

        # Determine torch dtype
        model_dtype_str = model_config.get("dtype", "bfloat16")
        if model_dtype_str == "bfloat16":
            model_dtype = torch.bfloat16
        elif model_dtype_str == "float16":
            model_dtype = torch.float16
        else:
            model_dtype = torch.float32
        print_rank_0(f"Using model dtype: {model_dtype}")

        # This line is wrong for Megatron, replace with actual Megatron model loading
        self.model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            config=hf_config,
            torch_dtype=model_dtype,
            attn_implementation="flash_attention_2" if model_config.get("use_flash_attention_2", False) else None,
            trust_remote_code=model_config.trust_remote_code,
        )
        print_rank_0("!!! WARNING: Using HF AutoModelForCausalLM as placeholder. Replace with Megatron model loading !!!")
        # --- End Temporary Placeholder ---


        # Apply LoRA if configured
        if model_config.get("lora_rank", 0) > 0:
            print_rank_0(f"Applying LoRA with rank {model_config.lora_rank}")
            # Determine target modules for Megatron Llama (needs inspection of the model structure)
            # Example: target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
            # Use target_modules from config or default Llama ones
            default_llama_targets = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            target_modules = model_config.get("target_modules", default_llama_targets)
            print_rank_0(f"LoRA target modules: {target_modules}")

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=model_config.lora_rank,
                lora_alpha=model_config.lora_alpha,
                target_modules=target_modules,
                lora_dropout=model_config.get("lora_dropout", 0.0),
                bias="none",
                # modules_to_save: Add if training embeddings/lm_head with LoRA
                # modules_to_save=["embed_tokens", "lm_head"] if model_config.get("lora_train_embedding") else None,
            )

            # Ensure input grads are enabled for LoRA (PEFT usually handles this)
            # self.model.enable_input_require_grads()

            self.model = get_peft_model(self.model, lora_config)
            print_rank_0("LoRA applied.")
            if self.data_parallel_rank == 0:
                self.model.print_trainable_parameters()

        # --- Megatron Model Parallel Wrapping (Placeholder) ---
        # Megatron usually handles DDP internally or requires explicit wrapping after model creation/LoRA application.
        # Example:
        # self.model = DDP(self.model, data_parallel_group=self.data_parallel_group) # Wrap the potentially PEFT-modified model
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
        # --- Megatron Optimizer (Placeholder) ---
        # Use Megatron's optimizer setup if available/required
        # optimizer_grouped_parameters = get_optimizer_param_groups(self.model) # Get params respecting DP/MP
        # self.optimizer = get_optimizer(optimizer_grouped_parameters, optim_config) # Pass Megatron optim config
        # --- End Megatron Optimizer ---

        # --- Standard PyTorch Optimizer (Use if compatible with Megatron's DDP/MP) ---
        # Filter parameters that require gradients
        params_to_optimize = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = optim.AdamW(
            params_to_optimize,
            lr=optim_config.lr,
            betas=tuple(optim_config.betas), # e.g., [0.9, 0.95] -> (0.9, 0.95)
            weight_decay=optim_config.weight_decay,
            eps=optim_config.get("eps", 1e-8) # AdamW epsilon
        )
        # --- End Standard PyTorch Optimizer ---
        print_rank_0(f"Optimizer built: {type(self.optimizer).__name__}")


        # Build Learning Rate Scheduler
        print_rank_0("Building LR scheduler...")
        # Calculate steps considering gradient accumulation if Megatron uses it
        # grad_accum_steps = self.args.gradient_accumulation_steps if hasattr(self.args, 'gradient_accumulation_steps') else 1
        grad_accum_steps = 1 # Assume 1 for now
        self.steps_per_epoch = len(self.train_dataloader) // grad_accum_steps
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs
        num_warmup_steps = int(self.total_steps * optim_config.warmup_steps_ratio)
        print_rank_0(f"Grad accum steps: {grad_accum_steps}, Steps per epoch: {self.steps_per_epoch}")
        print_rank_0(f"Total steps: {self.total_steps}, Warmup steps: {num_warmup_steps}")

        # --- Megatron LR Scheduler (Placeholder) ---
        # self.lr_scheduler = get_learning_rate_scheduler(self.optimizer, optim_config) # Pass Megatron scheduler config
        # --- End Megatron LR Scheduler ---

        # --- Standard Scheduler (from verl.utils) ---
        scheduler_type = optim_config.get("lr_scheduler", "cosine")
        if scheduler_type == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=self.total_steps
            )
        # Add other scheduler types (e.g., 'wsd') if needed, copying from fsdp_sft_trainer
        elif scheduler_type == "wsd":
             from verl.utils.torch_functional import get_wsd_schedule_with_warmup
             self.lr_scheduler = get_wsd_schedule_with_warmup(
                 optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps
             )
        else:
            raise ValueError(f"Unsupported lr_scheduler type: {scheduler_type}")
        # --- End Standard Scheduler ---
        print_rank_0(f"LR Scheduler built: {scheduler_type}")


    def _build_tracking(self):
        """Initializes the tracking backend (e.g., WandB)."""
        if self.data_parallel_rank == 0 and self.config.trainer.get("logger", None):
            logger_backend = self.config.trainer.logger
            print_rank_0(f"Initializing tracking: {logger_backend}")
            # Convert OmegaConf to dict for logging if necessary
            try:
                log_config = OmegaConf.to_container(self.config, resolve=True)
            except: # Handle cases where OmegaConf might not be used
                log_config = dict(self.config)

            self.tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=logger_backend,
                config=log_config # Log the config
            )
        else:
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

        # --- Megatron Forward/Backward (Placeholder) ---
        # This might be encapsulated in a single function like `train_step` in Megatron
        # loss_dict = train_step(forward_step_func=self.model.forward, # Or a custom func
        #                        data_iterator=iter([batch]), # Needs adaptation
        #                        model=self.model, # Or list of models for pipeline
        #                        optimizer=self.optimizer,
        #                        lr_scheduler=self.lr_scheduler)
        # loss = loss_dict['lm_loss'] # Example key
        # grad_norm = loss_dict['grad_norm'] # Example key
        # --- End Megatron Forward/Backward ---

        # --- Standard PyTorch Forward/Backward (Use if compatible) ---
        # Forward pass
        # Megatron models might require specific attention mask formats or position ids
        # Example:
        # attention_mask, position_ids = get_ltor_masks_and_position_ids(
        #     data=input_ids,
        #     eod_token=self.tokenizer.eod, # Example
        #     reset_position_ids=self.args.reset_position_ids,
        #     reset_attention_mask=self.args.reset_attention_mask,
        #     eod_mask_loss=self.args.eod_mask_loss)

        # Use autocast for mixed precision if configured (e.g., bfloat16)
        model_dtype = self.model.dtype # Get dtype from model
        with torch.autocast(device_type='cuda', dtype=model_dtype, enabled=(model_dtype != torch.float32)):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # position_ids=position_ids, # If needed by the model
                use_cache=False # Ensure no caching during training
            )
            logits = outputs.logits
            # Ensure loss computation is in float32 for stability
            loss = self._compute_loss(logits.to(torch.float32), labels, loss_mask)

        # Backward pass
        self.optimizer.zero_grad()
        # Megatron might scale the loss here before backward (e.g., for grad accum)
        # scaled_loss = loss / grad_accum_steps # Example
        # scaled_loss.backward()
        loss.backward() # Standard backward

        # Gradient Clipping (Megatron might do this internally via optimizer wrapper)
        grad_norm = None
        clip_grad_config = self.config.optim.get("clip_grad", None)
        if clip_grad_config is not None:
             # Use torch.nn.utils.clip_grad_norm_ for standard optimizers
             # Megatron's optimizer might handle clipping itself.
             grad_norm = torch.nn.utils.clip_grad_norm_(
                 filter(lambda p: p.requires_grad, self.model.parameters()), # Clip only trainable params
                 clip_grad_config
             )

        # Optimizer step
        self.optimizer.step()
        # --- End Standard PyTorch Forward/Backward ---

        # LR Scheduler step (after optimizer step)
        self.lr_scheduler.step()

        # Synchronize loss across data parallel ranks
        # --- Megatron Loss Averaging (Placeholder) ---
        # reduced_loss = average_losses_across_data_parallel_group([loss])
        # --- End Megatron Loss Averaging ---
        # --- Standard PyTorch AllReduce ---
        reduced_loss = loss.detach().clone()
        dist.all_reduce(reduced_loss, op=dist.ReduceOp.AVG)
        # --- End Standard PyTorch AllReduce ---

        lr = self.lr_scheduler.get_last_lr()[0]
        metrics = {"train/loss": reduced_loss.item(), "train/lr": lr}
        if grad_norm is not None:
            # Check if grad_norm is a tensor before calling item()
            metrics["train/grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

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

    def evaluate(self):
        """Runs evaluation on the validation dataset."""
        print_rank_0("Running validation...")
        self.model.eval()
        all_val_losses = []
        # Use try-finally to ensure model is set back to train mode
        try:
            val_iter = iter(self.val_dataloader)
            val_progress_bar = tqdm(
                range(len(self.val_dataloader)),
                disable=(self.data_parallel_rank != 0),
                desc="Validation"
            )

            for _ in val_progress_bar:
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
        print_rank_0(f"Validation finished. Average Loss: {avg_val_loss:.4f}")
        return {"val/loss": avg_val_loss}


    def fit(self):
        """Main training loop."""
        print_rank_0("Starting training...")
        global_step = 0
        # Calculate total training steps if not specified
        if self.config.trainer.get("total_training_steps") is None:
            self.total_training_steps = self.steps_per_epoch * self.config.trainer.total_epochs
        else:
            self.total_training_steps = self.config.trainer.total_training_steps
        print_rank_0(f"Total training steps: {self.total_training_steps}")

        # --- Load Checkpoint (Placeholder) ---
        # If resuming, load checkpoint here using Megatron's load_checkpoint
        # iteration = load_checkpoint(self.model, self.optimizer, self.lr_scheduler, ...)
        # global_step = iteration * grad_accum_steps # Adjust global step
        # --- End Load Checkpoint ---


        for epoch in range(self.config.trainer.total_epochs):
            print_rank_0(f"Starting Epoch {epoch+1}/{self.config.trainer.total_epochs}")
            self.train_sampler.set_epoch(epoch) # Ensure proper shuffling for distributed training

            train_iter = iter(self.train_dataloader)
            progress_bar = tqdm(
                range(len(self.train_dataloader)), # Iterate through batches
                disable=(self.data_parallel_rank != 0),
                desc=f"Epoch {epoch+1}"
            )

            for step_in_epoch in progress_bar:
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

                # Validation
                # Use >= 1 to ensure validation happens at least once if eval_steps is 1
                if global_step % self.config.trainer.eval_steps == 0 and self.config.trainer.eval_steps >= 1:
                    val_metrics = self.evaluate()
                    if self.tracking:
                        self.tracking.log({**val_metrics, "global_step": global_step})

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
            self.tracking.finish()


    def save_checkpoint(self, step, final=False):
        """Saves a checkpoint (model weights, tokenizer, potentially optimizer state)."""
        tag = "final" if final else f"global_step_{step}"
        # Use default_local_dir from config, default to ./checkpoints if not set
        local_save_dir = self.config.trainer.get("default_local_dir", "./checkpoints")
        save_path = os.path.join(local_save_dir, tag)
        print_rank_0(f"Saving checkpoint to {save_path}...")

        # --- Megatron Checkpoint Saving (Placeholder) ---
        # Megatron has specific checkpointing logic to handle model/data/pipeline parallelism.
        # It might save sharded states across ranks.
        # Example:
        # save_checkpoint(iteration=step, # Megatron uses iteration count
        #                 model=self.model,
        #                 optimizer=self.optimizer,
        #                 lr_scheduler=self.lr_scheduler,
        #                 args=self.args) # Pass Megatron args
        # The actual save path might be determined by Megatron's args (e.g., args.save).
        # Need to ensure consistency between `save_path` and Megatron's internal path.
        # --- End Megatron Checkpoint Saving ---

        # --- PEFT LoRA Adapter Saving (Use if LoRA is enabled) ---
        # This saves only the adapter weights, typically on rank 0.
        if isinstance(self.model, PeftModel):
            if self.data_parallel_rank == 0:
                os.makedirs(save_path, exist_ok=True)
                try:
                    self.model.save_pretrained(save_path)
                    self.tokenizer.save_pretrained(save_path)
                    # Optionally save optimizer state if needed for LoRA fine-tuning resume
                    # torch.save(self.optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
                    # torch.save(self.lr_scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))
                    print_rank_0(f"LoRA adapter and tokenizer saved to {save_path}")
                except Exception as e:
                    print_rank_0(f"ERROR: Failed to save LoRA checkpoint to {save_path}: {e}")
        # --- End PEFT LoRA Adapter Saving ---
        else:
            # --- Standard Full Model Saving (Placeholder for non-LoRA Megatron) ---
            # If not using LoRA, need to implement Megatron's full model saving.
            # This might involve gathering parameters on rank 0 or saving shards.
            # Megatron's `save_checkpoint` function should handle this.
            if self.data_parallel_rank == 0:
                 os.makedirs(save_path, exist_ok=True)
                 # Placeholder: self.model.save_pretrained(save_path) # If HF compatible save exists for the Megatron model
                 self.tokenizer.save_pretrained(save_path)
                 # Save optimizer/scheduler state for full model resume
                 # torch.save(self.optimizer.state_dict(), os.path.join(save_path, "optimizer.pt"))
                 # torch.save(self.lr_scheduler.state_dict(), os.path.join(save_path, "scheduler.pt"))
                 print_rank_0(f"Full model (placeholder) and tokenizer saved to {save_path}")
            # --- End Standard Full Model Saving ---

        # Ensure all ranks have finished local saving before potentially uploading
        dist.barrier()

        # Upload to HDFS if configured
        hdfs_dir = self.config.trainer.get("default_hdfs_dir", None)
        if self.data_parallel_rank == 0 and hdfs_dir:
            hdfs_target_path = os.path.join(hdfs_dir, tag) # Save with the same tag structure
            print_rank_0(f"Copying checkpoint from {save_path} to HDFS: {hdfs_target_path}")
            try:
                # Ensure parent HDFS directory exists
                parent_hdfs_dir = os.path.dirname(hdfs_target_path)
                if parent_hdfs_dir != hdfs_dir: # Avoid making root dir if target is directly under hdfs_dir
                     hdfs_io.makedirs(parent_hdfs_dir, exist_ok=True)
                # Copy the directory content
                hdfs_io.copy(src=save_path, dst=hdfs_target_path, dirs_exist_ok=True) # Use target path directly
                print_rank_0("Checkpoint copied to HDFS.")
            except Exception as e:
                print_rank_0(f"ERROR: Failed to copy checkpoint to HDFS: {e}")

        # Final barrier after potential HDFS copy
        dist.barrier()


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