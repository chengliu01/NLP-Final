import requests
import json
import logging
import copy
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

config = {
    "api_key": "chengliu",
    "base_url": "http://xxxxxxxxx:8081/v1"
}

streaming = False

def get_localllm_response(model, messages, max_tokens=None, nth_generation=0, is_safe=False, temperature=0.3, top_k=5, top_p=0.9, base_url=None):
    from gca_evaluation.utils import logger, num_tokens_from_string, get_response_with_safe

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    
    merged_messages = []
    for message in messages:
        if message['role'] == 'user' and merged_messages and merged_messages[-1]['role'] == 'user':
            merged_messages[-1]['content'] += message['content']
        else:
            merged_messages.append(copy.deepcopy(message))
    
    messages = merged_messages
    config["base_url"] = base_url

    try:
        url = f"{config['base_url']}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config['api_key']}"
        }
        
        request_params = {
            "model": model,
            "messages": messages,
            "temperature": temperature if nth_generation == 0 else 0.7,
            "top_k": top_k,
            "top_p": top_p
        }
        
        if max_tokens is not None:
            request_params["max_tokens"] = max_tokens
        response = requests.post(url, headers=headers, json=request_params)
        response_json = response.json()
        if "choices" in response_json and len(response_json["choices"]) > 0 \
            and "message" in response_json["choices"][0]:
            return response_json["choices"][0]["message"]["content"]
        else:
            logger.error(f"Invalid response format: {response_json}")
            return None
                
    except Exception as e:
        logger.error(f'Prompt: {messages[-1]["content"][:500]}')
        logger.error(f"Error in get_localllm_response: {str(e)} from model {model}")
        
        if ("content_filter" in str(e) or "invalid_prompt" in str(e)) and not is_safe:
            logger.warning("Content filter triggered. Retrying with safe message.")
            safe_messages = get_response_with_safe(messages)
            return get_localllm_response(model, safe_messages, max_tokens, nth_generation + 1, is_safe=True)
        
        try:
            if hasattr(response, 'text'):
                logger.error(f"Response: {response.text}")
            else:
                logger.error(f"Response: {response}")
        except Exception as e:
            logger.error(f"Could not print response: {e}")
        
        if isinstance(messages, list) and len(messages) > 0:
            logger.error(f"Number of input tokens: {num_tokens_from_string(messages[0]['content'])}")
        
        import traceback
        traceback.print_exc()
        return None

def __test__():
    method = "fixed"
    pre_nums = 0
    from model.example import load_example_messages
    from gca_evaluation.utils import parse_reasoning_response

    test_messages = load_example_messages(method, pre_nums, use_random=True, seed=2)
    # print("Test messages: {}".format(test_messages[0]['content']+"\n\n"))
    # import pdb; pdb.set_trace()
    model = "CogDual-Qwen25-7B-Instruct"
    model = "CoSER-Llama-3.1-70B-Instruct"
    model = "CoT-Qwen2.5-7B-Instruct"
    # model = "DS-Distill-Llama3.1-8B"
    model = "Llama3.1-8B-Instruct"

    # print(test_messages)
    del test_messages[0]['reference']
    
    test_messages[-1]["content"] += "\n<think>"

    response = get_localllm_response(
        model=model,
        messages=test_messages,
        temperature=0.0,
        top_k=1,
        top_p=0.9
    )
    # answer, _ = parse_reasoning_response(response, tag="cognitive", is_cogdual=True) 
    
    print(f"Model: {model}")
    print(f"Response:\n{response}")


if __name__ == "__main__":
    __test__()