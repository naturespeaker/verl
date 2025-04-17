set -x  # 在执行脚本时打印出每一条被执行的命令，便于调试。

gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet
math_train_path=$HOME/data/math/train.parquet
math_test_path=$HOME/data/math/test.parquet

train_files="['$gsm8k_train_path', '$math_train_path']"
test_files="['$gsm8k_test_path', '$math_test_path']"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \       # 指定使用 GAE (Generalized Advantage Estimation) 算法来估计 PPO 中的优势函数 (Advantage)。GAE 是一种常用的减少 PPO 算法方差的技术。
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=4096 \        # 训练时的批次大小，这里指的是 token 数量，而不是样本数量。这意味着每个训练批次包含大约 4096 个 token。
    data.max_prompt_length=4096 \       # 输入提示 (Prompt) 的最大 token 长度。
    data.max_response_length=4096 \     # 模型生成响应 (Response) 的最大 token 长度。
    data.filter_overlong_prompts=True \     # 过滤掉超过 max_prompt_length 的提示。
    data.truncation='error' \           # 如果输入或输出超过最大长度，则报错（而不是截断）。
    actor_rollout_ref.model.path=Qwen/Qwen2-7B-Instruct \   # 指定 Actor 和 Reference 模型的基础模型路径，这里是 Hugging Face Hub 上的 Qwen/Qwen2-7B-Instruct。
    actor_rollout_ref.actor.optim.lr=1e-6 \     # Actor 模型优化器的学习率。
    actor_rollout_ref.model.use_remove_padding=True \   # 可能启用了一种优化，在计算中移除 padding token，以提高效率。
    actor_rollout_ref.model.enable_gradient_checkpointing=True \    # 启用梯度检查点。这是一种用计算时间换显存的技术，允许训练更大的模型或使用更大的批次大小。
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \               # 在 PPO 的优化步骤中，将收集到的数据分成更小的 mini-batch 进行梯度更新，这里每个 mini-batch 包含 512 个样本（Prompt-Response 对）。
    actor_rollout_ref.actor.use_dynamic_bsz=True \          # 启用动态批处理大小。这可能意味着根据序列长度动态调整每个批次包含的样本数，以更好地利用 GPU。
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \   # (序列平衡关键参数) 在 PPO Actor 更新阶段，限制每个 GPU 处理的最大 token 总数。这有助于在不同 GPU 之间平衡负载，特别是当样本序列长度差异很大时。
    actor_rollout_ref.actor.fsdp_config.param_offload=False \   # FSDP (Fully Sharded Data Parallel) 相关配置。这里禁用了参数 (param) 和
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \   # 优化器状态 (optimizer) 的 CPU offload，意味着它们将保留在 GPU 显存中。
    actor_rollout_ref.actor.use_kl_loss=False \     # 在 Actor 的损失函数中不使用 KL 散度惩罚项。
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \    # 在 Rollout 阶段（生成数据）使用 2 路张量并行 (Tensor Parallelism)。这意味着模型的权重会被切分到 2 个 GPU 上。
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \  # SGLang 推理时使用的 GPU 显存比例。
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24000 \    # (序列平衡关键参数) 在 Rollout 阶段计算 log probabilities 时，限制每个 GPU 处理的最大 token 总数，同样是为了负载平衡。
    # 这部分配置 PPO 中的 Critic 模型（价值网络），用于评估状态的价值。
    critic.optim.lr=1e-5 \  # Critic 模型优化器的学习率。
    critic.model.use_remove_padding=True \      # Critic 模型也启用移除 padding 的优化。
    critic.model.path=Qwen/Qwen2-7B-Instruct \      # Critic 模型也使用 Qwen/Qwen2-7B-Instruct 作为基础模型进行初始化。
    critic.model.enable_gradient_checkpointing=True \   # Critic 模型也启用梯度检查点。
    critic.ppo_max_token_len_per_gpu=98304 \        # (序列平衡关键参数) 在 Critic 更新阶段，限制每个 GPU 处理的最大 token 总数。这个值通常比 Actor 的大，因为 Critic 的计算相对简单。
    critic.model.fsdp_config.param_offload=False \      #  Critic 模型的 FSDP 配置，同样禁用了 offload。
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    # 训练器相关
    trainer.critic_warmup=0 \   # Critic 模型没有预热步骤。
    trainer.logger=['console','wandb'] \    # 使用控制台 (console) 和 Weights & Biases (wandb) 来记录训练日志和实验指标。
    trainer.project_name='verl_example_gsm8k' \     # WandB 项目名称。
    trainer.experiment_name='qwen2-7b_function_rm_bsz8k_p4k_r4k_seq_packing' \  # WandB 实验名称，包含了模型、数据集、批大小、序列长度和序列平衡等信息。
    trainer.n_gpus_per_node=8 \     # 每个计算节点使用 8 个 GPU。
    trainer.val_before_train=False \    # 不在训练开始前进行验证。
    trainer.nnodes=1 \  # 使用 1 个计算节点进行训练。结合 n_gpus_per_node=8，表示总共使用 8 个 GPU。
    trainer.save_freq=-1 \  # 不按固定的频率保存模型检查点（可能只在训练结束时保存）。
    trainer.test_freq=5 \   # 每训练 5 个 epoch 后进行一次测试（验证）。
    trainer.total_epochs=15 $@  # 总共训练 15 个 epoch。
    # $@ (Line 51): 将运行此 shell 脚本时传递给脚本的所有额外参数追加到 python3 命令的末尾。
    # 这允许你在命令行动态修改配置，例如 bash run_qwen2-7b_sglang_seq_balance.sh trainer.total_epochs=10。
