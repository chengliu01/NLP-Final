from email.policy import strict
import os
import json
import requests
import logging
import traceback
import copy, re
from typing import Dict, List, Optional, Any
import sys, openai
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

config_fp = "./CoSER/config/config_api_0425.json"

with open(config_fp, 'r') as f:
	config = json.load(f)


def get_chatgpt_response(model, messages, max_tokens=None, nth_generation=0, temperature=0.0):
	
    from gca_evaluation.utils import (logger, num_tokens_from_string, get_response_with_safe)

    if "o1" in model and messages[0]["role"] == "system":
        messages[0]["role"] = "user"

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
		
    if model.startswith('claude') and messages and messages[0]['role'] == 'system': messages[0]['role'] = 'user'	

    merged_messages = []
    for message in messages:
        if message['role'] == 'user' and merged_messages and merged_messages[-1]['role'] == 'user':
            merged_messages[-1]['content'] += message['content']
        else:
            merged_messages.append(copy.deepcopy(message))

    messages = merged_messages
    response = None

    try:
        os.environ['OPENAI_API_KEY'] = config['api_key']
        os.environ['OPENAI_BASE_URL'] = config['base_url']
        client = openai.OpenAI()
        if max_tokens is None: max_tokens = 4096 # 8196 评估的 tokens
        request_params = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature if nth_generation == 0 else 1.0,
            "timeout": 180
        }

        if "o1" in model:
            del request_params["max_tokens"]
            del request_params["temperature"]

        completion = client.chat.completions.create(
            **request_params
        )
        response = completion.choices[0].message.content
        
    except Exception as e:
        import traceback 

        logger.error(f'Prompt: {messages[-1]["content"][:500]}')
        logger.error(f"Error in get_response: {str(e)} from model {model}")
        strict_level = nth_generation

        if ("content_filter" in str(e) or "invalid_prompt" in str(e)) and strict_level<=3:
            logger.warning(f"Content filter triggered. Retrying with safe message., strict_level: {strict_level}")
            safe_messages = get_response_with_safe(messages, strict_level=strict_level)
            # import pdb; pdb.set_trace()
            return get_chatgpt_response(model, safe_messages, max_tokens, nth_generation + 1)

        try:
            if hasattr(response, 'text'):
                logger.error(f"Response: {response.text}")
            else:
                logger.error(f"Response: {response}")
        except Exception as e:
            logger.error(f"Could not print response: {e}")

        logger.error(f"Number of input tokens: {num_tokens_from_string(messages[0]['content'])}")

        traceback.print_exc()
        return ""
    
    return response


def __test__():

    """
    """
    from gca_evaluation.utils import parse_reasoning_response
    from example import load_example_messages

    model = "o1-preview"

    method_list = ["fixed"]
    response_list = []
    pre_nums = 0
    
    for i, method in enumerate(method_list):
        test_messages = load_example_messages(
            method=method, pre_nums=pre_nums, use_random=True, seed=45
        )
        if i == 0:
            print(f"Prompt:\n"+test_messages[0]["content"])

        del test_messages[0]["reference"]
        # import pdb; pdb.set_trace()
        
        response = get_chatgpt_response(
            model=model,
            messages=test_messages,
            max_tokens=512
        )
        response_list.append(response)
        print(f"\n\n==={method} response===\n\n{response}")


if __name__ == "__main__":
    # __test__()
    __test_safe()