# MDS5110 Final Project: Contextual Awareness and Self-Cognition for Dynamic Character Consistency

This repository contains the code and resources for the final project of MDS5110 at CUHKSZ, titled **"Contextual Awareness and Self-Cognition for Dynamic Character Consistency"**.

## Project Structure

- **Model Training:**  
  The experimental models are reproduced based on the following training scripts:
  - `verl/examples/sft/multiturn/run_qwen_7b.sh`
  - `verl/examples/sft/multiturn/run_llama_8b.sh`

- **Model Deployment:**  
  The trained models are deployed using:
  - `vllm_server.sh`

- **Model Evaluation:**  
  The model performance is evaluated on CoSER using:
  - `CoSER/gca_evaluation/main.sh`