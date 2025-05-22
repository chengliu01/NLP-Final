MODEL_NAME=your_model

echo "Starting script at: $(date)"
start_time=$(date +%s)

python gca_evaluation/main.py \
    --test_file data/test/test_set.json \
    --actor_model ${MODEL_NAME} \
    --judge_model gpt-4o \
    --nsp_model gpt-4o \
    --env_model gpt-4o \
    --num_workers 4 \
    --use_reasoning \
    --cognitive_method sft_cognitive

end_time=$(date +%s)
echo "Finished script at: $(date)"

duration=$((end_time - start_time))
echo "Total execution time: ${duration} seconds"