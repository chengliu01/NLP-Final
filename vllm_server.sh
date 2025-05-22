LLAMA_PATH=your_model_path

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m vllm.entrypoints.openai.api_server --host your_ip \
    --port 8081 \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.9 \
    --model ${LLAMA_PATH} \
    --served-model-name your_save_model_name \
    --dtype float16 \
    --api-key chengliu \
    --max-model-len 8192