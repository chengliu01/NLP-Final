set -x

NET_TYPE="low"
export NCCL_IB_TIMEOUT=24
if [[ "${NET_TYPE}" = "low" ]]; then
    export NCCL_SOCKET_IFNAME=eth1
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_HCA=mlx5_2:1,mlx5_2:1
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
    export NCCL_DEBUG=INFO
else
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
    export NCCL_SOCKET_IFNAME=bond1
    export UCX_NET_DEVICES=bond1
    export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
    export NCCL_COLLNET_ENABLE=0
    export SHARP_COLL_ENABLE_SAT=0
    export NCCL_NET_GDR_LEVEL=2
    export NCCL_IB_QPS_PER_CONNECTION=4
    export NCCL_IB_TC=160
    export NCCL_PXN_DISABLE=1
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
WORKSPACE=/apdcephfs_cq10/share_1567347/share_info/sorenliu/code/rl_code/verl
cd ${WORKSPACE}

# MODEL_PATH=/apdcephfs_cq10/share_1567347/share_info/sorenliu/llm_model/base_models/Qwen2.5-7B-Instruct

MODEL_PATH=/apdcephfs_cq10/share_1567347/share_info/llm_models/Qwen2.5-0.5B-Instruct

# export USE_FLASH_ATTENTION=0
export VLLM_ATTENTION_BACKEND=XFORMERS

bash /apdcephfs_cq10/share_1567347/share_info/sorenliu/code/script/unoccupy_single_node.sh

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${WORKSPACE}/data/gsm8k/train.parquet \
    data.val_files=${WORKSPACE}/data/gsm8k/test.parquet \
    data.train_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen2_7b_function_rm' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=2 $@


bash /apdcephfs_cq10/share_1567347/share_info/sorenliu/code/script/taiji_gpu/occupy_single_node.sh