set -x
nproc_per_node=8

conda activate verl_sft
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

WORKSPACE=verl

cd ${WORKSPACE}
DATA_DIR=${WORKSPACE}/data/coser_trainset/trainset

EPOCH=2
batch_size=64
lr=1e-5
MAX_LEN=10240
DATA_NAME=train_mix_0508_cogdual

TRAIN_PATH=${DATA_DIR}/${DATA_NAME}.parquet
VAL_PATH=${WORKSPACE}/data/coser_trainset/test_example/test.parquet

EXPERIMENT_NAME=multiturn-sft-${DATA_NAME}_epoch${EPOCH}_bs${batch_size}_lr${lr}_maxlen${MAX_LEN}

MODEL_PATH="Your model path"
SAVE_PATH="your save dir"/${EXPERIMENT_NAME}


if [ ! -d ${SAVE_PATH} ]; then
    mkdir -p ${SAVE_PATH}
fi


torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=${TRAIN_PATH} \
    data.val_files=${VAL_PATH} \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.train_batch_size=${batch_size} \
    data.micro_batch_size=1 \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=${MAX_LEN} \
    model.partial_pretrain=${MODEL_PATH} \
    model.enable_gradient_checkpointing=true \
    model.fsdp_config.offload_params=true \
    model.fsdp_config.cpu_offload=true \
    trainer.default_local_dir=${SAVE_PATH} \
    trainer.project_name=multiturn-sft \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.logger=['console'] \
    trainer.total_training_steps=null \
    trainer.total_epochs=${EPOCH} \
    trainer.default_hdfs_dir=null $@ \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true \
    data.truncation=right \
    optim.lr=${lr} \
    optim.lr_scheduler=cosine \
    optim.weight_decay=0.01 \
    optim.warmup_steps_ratio=0.05 \
    trainer.save_steps=-1 2>&1 | tee ${SAVE_PATH}/train.log