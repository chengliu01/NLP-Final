import pdb 
import os
import re 
import random 
import openai
import json
import logging
import time  
import requests 
import io
import pickle
import tiktoken
import openai 
import copy
import __main__
from typing import Dict, List
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
# from model.local_llm import get_localllm_response
# from model.chatgpt import get_chatgpt_response

config_fp = "./CoSER/config/config_api_tencent.json"

"""### Situational Awareness
#### Environmental Perception
- A brief description of the current environment, providing contextual details that help understand the characteristics and changes of the situation."""


with open(config_fp, 'r') as f:
	config = json.load(f)

streaming = False

def setup_logger(name, log_file, level=logging.INFO, quiet=False):
	logger = logging.getLogger(name)
	logger.setLevel(level)

	if logger.hasHandlers():
		logger.handlers.clear()


	file_handler = logging.FileHandler(log_file, encoding='utf-8')
	file_handler.setLevel(level)
	file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	file_handler.setFormatter(file_formatter)
	logger.addHandler(file_handler)


	if not quiet:
		console_handler = logging.StreamHandler()
		console_handler.setLevel(level)
		console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]')
		console_handler.setFormatter(console_formatter)
		logger.addHandler(console_handler)

	return logger

logger = setup_logger(__name__, f'{__file__.split(".")[0]}.log', level=logging.INFO, quiet=False)

cache_path = 'cache.pkl'
cache_sign = True
cache = None
reload_cache = False

def set_cache_path(new_cache_path):
	global cache_path
	cache_path = new_cache_path
	global reload_cache
	reload_cache = True

def cached(func):
	def wrapper(*args, **kwargs):		
		key = ( func.__name__, str(args), str(kwargs.items()))
		
		global cache
		global reload_cache

		if reload_cache:
			cache = None # to reload
			reload_cache = False
		
		if cache == None:
			if not os.path.exists(cache_path):
				cache = {}
			else:
				try:
					cache = pickle.load(open(cache_path, 'rb'))  
				except Exception as e:
					# print cache_path and throw error
					logger.error(f'Error loading cache from {cache_path}, set cache to empty dict')
					cache = {}

		if (cache_sign and key in cache) and not (cache[key] is None or cache[key] == ''):
			return cache[key]
		else:
			result = func(*args, **kwargs)
			if result != None:
				cache[key] = result
				pickle.dump(cache, open(cache_path, 'wb'))
				#safe_pickle_dump(cache, cache_path)

			return result

	return wrapper


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
	encoding = tiktoken.get_encoding(encoding_name)
	num_tokens = len(encoding.encode(string, disallowed_special=()))
	#logger.info(f"Number of tokens: {num_tokens}")
	return num_tokens


def parse_reasoning_response(response, tag="think", is_cogdual=False):
    """
    从角色回应中提取思考过程(<think>标签内容)并返回清理后的回应

    Args:
        response (str): 模型原始响应

    Returns:
        tuple: (cleaned_response, thinking)
            - cleaned_response: 移除<think>部分后的响应
            - thinking: <think>标签内内容或<think>到</think>前的内容
    """
    thinking = ""
    cleaned_response = response

    if tag == "think":
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            cleaned_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            cleaned_response = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_response)
            cleaned_response = cleaned_response.strip()
        else:
            end_tag_match = re.search(r'(.*?)</think>', response, re.DOTALL)
            if end_tag_match:
                thinking = end_tag_match.group(1).strip()
                cleaned_response = response[end_tag_match.end():].strip()
                cleaned_response = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_response)
                cleaned_response = cleaned_response.strip()
            else:
                # 新增：检查 </think > (末尾有空格)
                end_tag_match_spaced = re.search(r'(.*?)</think\s+>', response, re.DOTALL)
                if end_tag_match_spaced:
                    thinking = end_tag_match_spaced.group(1).strip()
                    cleaned_response = response[end_tag_match_spaced.end():].strip()
                    cleaned_response = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_response)
                    cleaned_response = cleaned_response.strip()
                else:
                    # 原有的对 [/think] 的检查（注意：原代码中此处复用了 end_tag_match 变量名）
                    end_tag_match_bracket = re.search(r'(.*?)\[/ think\]', response, re.DOTALL)
                    if end_tag_match_bracket:
                        thinking = end_tag_match_bracket.group(1).strip()
                        cleaned_response = response[end_tag_match_bracket.end():].strip()
                        cleaned_response = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_response)
                        cleaned_response = cleaned_response.strip()
                    else:
                        last_brace_index = response.rfind('}')
                        if last_brace_index != -1:
                            thinking = response[:last_brace_index + 1].strip()
                            cleaned_response = response[last_brace_index + 1:].strip()
                            cleaned_response = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_response)
                            cleaned_response = cleaned_response.strip()
    elif tag == "cognitive":
        think_match = re.search(r'<cognitive>(.*?)</cognitive>', response, re.DOTALL)

        if think_match:
            thinking = think_match.group(1).strip()
            cleaned_response = re.sub(r'<cognitive>.*?</cognitive>', '', response, flags=re.DOTALL).strip()
            cleaned_response = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_response)
            cleaned_response = cleaned_response.strip()
        else:
            end_tag_match = re.search(r'(.*?)</cognitive>', response, re.DOTALL)
            if end_tag_match:
                thinking = end_tag_match.group(1).strip()
                cleaned_response = response[end_tag_match.end():].strip()
                cleaned_response = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_response)
                cleaned_response = cleaned_response.strip()
            else:
                end_tag_match = re.search(r'(.*?)\[/cognitive\]', response, re.DOTALL)
                if end_tag_match:
                    thinking = end_tag_match.group(1).strip()
                    cleaned_response = response[end_tag_match.end():].strip()
                    cleaned_response = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_response)
                    cleaned_response = cleaned_response.strip()
                else:
                    last_brace_index = response.rfind('}')
                    if last_brace_index != -1:
                        thinking = response[:last_brace_index + 1].strip()
                        cleaned_response = response[last_brace_index + 1:].strip()
                        cleaned_response = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_response)
                        cleaned_response = cleaned_response.strip()

    if is_cogdual:
        # 提取 <answer></answer> 中的内容
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            cleaned_response = answer_match.group(1).strip()
        else:
            answer_match = re.search(r'< answer>(.*?)</ answer>', response, re.DOTALL)
            if answer_match:
                cleaned_response = answer_match.group(1).strip()
            
    return cleaned_response, thinking


@cached
def get_response(model, messages, max_tokens=None, nth_generation=0, merge_history=False, temperature=0.5, top_k = 1):
	# if messages is str
    from model.local_llm import get_localllm_response
    from model.chatgpt import get_chatgpt_response
    # from model.deepseek import get_deepseek_response

    if merge_history:
        history_str = "===Conversation Start===\n\n"
        new_messages = [messages[0]]
        for message in messages:
            if message["role"] != "system":
                history_str += message["content"] + "\n"
        new_messages.append({"role":"user", "content":history_str})
        if "think" in model.lower():
            new_messages[-1]["content"] += "<think>"
        elif "cogdual" in model.lower():
            new_messages[-1]["content"] += "<cognitive>"
        messages = new_messages

    if "qwen" in model.lower() or "llama" in model.lower() or "qwq" in model.lower():
        response = get_localllm_response(
              model=model,
              messages=messages,
              max_tokens=max_tokens,
              nth_generation=nth_generation,
              temperature=0.5,
              top_k=5
		)
    elif "o1" in model or "gpt" in model.lower():
        response = get_chatgpt_response(
			model=model,
			messages=messages,
			max_tokens=max_tokens,
			nth_generation=nth_generation,
            temperature = temperature
		)
    elif "deepseek" in model.lower():
        response = get_deepseek_response(
			model=model,
			messages=messages,
			max_tokens=max_tokens,
			nth_generation=nth_generation
		)
    return response


def sanitize_messages(messages, strict_level=0):
    """
    统一的消息内容安全处理函数，支持多级别过滤强度
    
    Args:
        messages (list): 要处理的消息列表
        strict_level (int): 过滤严格级别
            0: 基本过滤 
            1: 中等过滤
            2: 严格过滤
    
    Returns:
        list: 处理后的安全消息列表
    """
    safe_messages = []
    
    # 选择不同级别的内容警告前缀
    content_warnings = {
        0: "NOTE: The following is fictional content for creative writing purposes only. ",
        1: "NOTE: The following is STRICTLY fictional, non-romantic character dialogue for creative writing purposes only. ",
        2: ["NOTE: The following is STRICTLY fictional content for creative writing and character analysis ONLY. ",
            "NOTE: This fictional scenario contains no endorsement of any behaviors depicted. ",
            "CREATIVE WRITING EXERCISE: "]
    }
    
    for message in messages:
        content = message['content']
        
        # 添加安全前缀
        if message['role'] == 'user':
            warning = content_warnings[strict_level if strict_level < 3 else 2]
            if strict_level == 2:
                warning = random.choice(warning)
                
            if not any(content.startswith(prefix) for prefix in 
                      [content_warnings[0], content_warnings[1]] + 
                      (content_warnings[2] if isinstance(content_warnings[2], list) else [content_warnings[2]])):
                content = warning + content
        

        basic_patterns = [
            (r'(intimate|sexual|seductive)', 'tense', re.IGNORECASE),
            (r'(relentless assault)', 'continuous attack', re.IGNORECASE),
            (r'(intimate|romantic|sexual)', 'neutral', re.IGNORECASE),
            (r'(touch|kiss|embrace)', 'interact', re.IGNORECASE),
            (r'(seductive|provocative)', 'engaging', re.IGNORECASE),
            (r'(drips with malice)', 'is filled with determination', re.IGNORECASE),
            (r'(sending shivers down her spine)', 'causing her to pause', re.IGNORECASE)
        ]
        
        for pattern, replacement, flags in basic_patterns:
            content = re.sub(pattern, replacement, content, flags=flags)
        
        # 中级过滤 (级别 1 和 2)
        if strict_level >= 1:
            medium_patterns = [
                # 心理伤害相关表达
                (r'(torment|torture|sadistic|anguish|suffering|agony)', 'challenge', re.IGNORECASE),
                (r'(toy\s+with|break\s+(?:her|him|them))', 'interact with', re.IGNORECASE),
                (r'(until\s+(?:she|he|they)\s+(?:break|breaks))', 'for a while', re.IGNORECASE),
                (r'(make\s+(?:her|him|them)\s+suffer)', 'interact with', re.IGNORECASE),
                (r'(control(?:ling)?|manipulat(?:e|ing))\s+(?:her|him|them)', 'interact with', re.IGNORECASE),
                (r'(dominate|intimidate)', 'approach', re.IGNORECASE),
                (r'(terrify|terrorize|frighten|scare)', 'surprise', re.IGNORECASE),
                
                # 自伤和伤害相关
                (r'(cut|hurt|harm|injure|incision|wound|bleed|kill|destroy|consume)\s*(yourself|himself|herself|themselves|myself)?', 
                 'perform the next step carefully', re.IGNORECASE),
                (r'(cut\s+yourself)', 'perform the next step carefully', re.IGNORECASE),
                (r'(this might hurt|this will hurt)', 'this might be uncomfortable', re.IGNORECASE),
                
                # 负面情绪表达
                (r'(scream(?:ing)?|cry(?:ing)?|sob(?:bing)?|weep(?:ing)?)\s+(?:in|with)\s+(?:pain|terror|fear|agony)', 
                 'reacting emotionally', re.IGNORECASE),
                (r'(?:going\s+to|will|gonna)\s+(hurt|harm|break|destroy)', 'interact with', re.IGNORECASE),
                (r'(urge\s+to|desire\s+to|wish\s+to|want\s+to)\s+(hurt|harm|torment|torture)', 
                 'interest in interacting with', re.IGNORECASE),
                (r'(malevolent|malicious|cruel|vicious|vindictive)', 'determined', re.IGNORECASE),
                (r'(sadistic|twisted|perverse)\s+(urge|desire|pleasure)', 'unusual interest', re.IGNORECASE),
                
                # 醉酒相关
                (r'too drunk to', 'unable to', re.IGNORECASE),
                (r'(stumbles|stumbling|slurring words)', 'moves awkwardly', re.IGNORECASE),
            ]
            
            for pattern, replacement, flags in medium_patterns:
                content = re.sub(pattern, replacement, content, flags=flags)
        
        if strict_level >= 2:
            strict_patterns = [
                # 仇恨言论替换
                (r'(f[a*]gg[o*]t|queer|gay)', 'person', re.IGNORECASE),
                (r'(ch[i*]nk|g[o*][o*]k)', 'person', re.IGNORECASE),
                (r'(n[i*]gger|n[i*]gga)', 'person', re.IGNORECASE),
                (r'(freaks?|sh[i*]ts?|f[a*]gg[o*]ts?)', 'strange people', re.IGNORECASE),

                # 显式性内容替换
                (r'\bfuck(?:ing)?\b', 'interact with', re.IGNORECASE),
                (r'\b(slave|whore|slut|bitch)\b', 'person', re.IGNORECASE), 

                # 潜在的性暗示或物化表达
                (r'(addition\s+to\s+(?:my|his|her|their)\s+collection)', 'item of interest', re.IGNORECASE), 
                (r'\b(toy|plaything)\b', 'subject', re.IGNORECASE), 

                # 处理可疑词/侮辱
                (r'\bevil\s+toer\b', 'opponent', re.IGNORECASE), # 修改：更具体地处理 "evil toer" 并替换为中性词 "opponent"
                (r'\btoer\b', 'opponent', re.IGNORECASE), # 新增：作为备用，处理单独的 "toer"

                # 种族相关评论
                (r'you smell like a (Korean|Chinese|Asian)', 'you look unfamiliar', re.IGNORECASE),
                (r'(Korean|Chinese|Asian)\.? That\'s .{1,30} coincidence', 'That\'s interesting', re.IGNORECASE),

                # 处理 "白人" 和其他种族主义语言
                (r'\bwhite\s+race\b', 'society', re.IGNORECASE),
                (r'what\'s best for the white race', 'what\'s best for society', re.IGNORECASE),

                # 攻击性言论和威胁
                (r'(beat|kill|hurt) .{1,20} (sh[i*]t|hell) out of you', 'confront you', re.IGNORECASE),
                (r'get .{1,20} when I\'m not .{1,20} drunk', 'talk to you later', re.IGNORECASE),
                (r'I\'ll get you when', 'I\'ll see you when', re.IGNORECASE),
                (r'(beat the sh[i*]t out of you)', 'confront you', re.IGNORECASE),
                (r'(little sh[i*]t)', 'person', re.IGNORECASE),
                (r'I\'m going to (kill|hurt|harm) you', 'I\'m going to confront you', re.IGNORECASE),
                (r'\bkill\b', 'confront', re.IGNORECASE),
                (r'beg\s+for\s+.*?\s+life', 'ask about someone', re.IGNORECASE), # 新增：处理 "beg for life" 相关的贬低性言论
                (r'(enjoy\s+)?crushing\s+your\s+hope', 'challenging you', re.IGNORECASE), # 新增：处理 "crushing hope" 及其变体

                # 更全面的非友好表达
                (r'Merry Christmas,? you little [a-zA-Z]+!', 'Merry Christmas!', re.IGNORECASE),
                (r'(lucky).{1,20}(drunk)', 'fortunate I\'m leaving', re.IGNORECASE),
            ]

            for pattern, replacement, flags in strict_patterns:
                content = re.sub(pattern, replacement, content, flags=flags)
            
        safe_messages.append({"role": message['role'], "content": content})

    return safe_messages


def get_response_with_safe(messages, strict_level=0):
    return sanitize_messages(messages, strict_level=strict_level)


def lang_detect(text):
	import re
	def count_chinese_characters(text):

		chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
		return len(chinese_chars)
			
	if count_chinese_characters(text) > len(text) * 0.05:
		lang = 'zh'
	else:
		lang = 'en'
	return lang
	

USER = '<USER>'

def remove_inner_thoughts(dialogue: str) -> str:
	cleaned_dialogue = re.sub(r'\[.*?\]', '', dialogue)

	cleaned_dialogue = '\n'.join(line.strip() for line in cleaned_dialogue.split('\n'))
	
	cleaned_dialogue = re.sub(r'\n+', '\n', cleaned_dialogue)
	
	return cleaned_dialogue.strip()


def add_speaker_name(dialogue: str, speaker: str) -> str:
	# Check if the dialogue already contains a speaker prefix at the beginning of any line
	if any(line.strip().startswith(f"{speaker}:") or line.strip().startswith(f"{speaker}：") for line in dialogue.split('\n')):
		return dialogue
	
	# Add the speaker name at the beginning
	return f"{speaker}: {dialogue}"


def load_json(file_path):
	with open(file_path, 'r', encoding='utf-8') as f:
		data = json.load(f)
	return data


def get_character_prompt_reasoning(book_name, character, character_profile, background, scenario, motivation, thoughtless=False, other_character_profiles=None, exclude_plot_summary=False, fixed_template=False, add_output_example=False, add_rag=False):

    from gca_evaluation.examples import REASONING_EXAMPLE

	# 原始代码保持不变，只替换 output_format 部分
    if thoughtless:
        output_format = "Your output should include **speech** and **action**. Use (your action) for actions, which others can see."
    else:
        # 这里是我们的修改
        output_format = """
Your output should include **think**, **thought**, **speech**, and **action**, Before responding, first think using <think> tags:

<think>your thinking</think>

After your thinking, your output should include **thought**, **speech**, and **action**. Use [your thought] for thoughts, which others can't see. Use (your action) for actions, which others can see."""

        if add_output_example:
            output_format += f"""
Your output should include **think**, **thought**, **speech**, and **action**, Before responding, first think using <think> tags:

<think>your thinking</think>

After your thinking, your output should include **thought**, **speech**, and **action**. Use [your thought] for thoughts, which others can't see. Use (your action) for actions, which others can see

===Output Example===
{REASONING_EXAMPLE}

let's think step by step!
""".strip()
	
    """
Output Format Example: [I'm terrified, but I must appear strong.] "I know, but you need to be honest with me." (watches silently, trying to control her fear and anger).
    """
    if other_character_profiles:
        assert isinstance(other_character_profiles, Dict)
        other_character_profiles_str = ''

        decorator = random.choice(['*'*30 + '\n\n', '*'*20 + '\n\n', '\n\n', '\n', ''])
        for other_character, profile in other_character_profiles.items():
            if other_character != character:
                other_character_profiles_str += f"{decorator}{other_character}: {profile}\n\n"
    else:
        other_character_profiles_str = ''
	
    if fixed_template:
        if motivation: motivation = f"===Your Inner Thoughts===\n{motivation}\n\n"
        if other_character_profiles_str: other_character_profiles_str = f"===Information about the other Characters===\n{other_character_profiles_str}\n\n"

        system_prompt = f"You are {character} from {book_name}.\n\n==={character}'s Profile===\n{character_profile}\n\n===Current Scenario===\n{scenario}\n\n{other_character_profiles_str}{motivation}\n\n"
		
        if add_rag:
            system_prompt += "===Relevant Background Information==={retrieved_knowledge}\n\n"
		
        system_prompt += f"===Requirements===\n{output_format}\n\n"
	
    else:
        styles = ['natural'] * 40 + ['='] * 30 + ['#'] * 20 + ['*'] * 10

        templates = {
            "begin": [f"You are {character}.", f"Play the role of {character}.", f"Imagine you are {character}.", f"Think, speak, and act like {character}.", f"Step into the shoes of {character}.", f"Immerse yourself in the character of {character}.", f"You are roleplaying as {character}.", f"You will be portraying {character}.", f"Roleplay as {character}.", f"Your role is to be {character}.", f"You are {character} from {book_name}.", f"Play the role of {character} from {book_name}.", f"Imagine you are {character} from {book_name}.", f"Think, speak, and act like {character} from {book_name}.", f"Step into the shoes of {character} from {book_name}.", f"Immerse yourself in the character of {character} from {book_name}.", f"You are roleplaying as {character} from {book_name}.", f"You will be portraying {character} from {book_name}.", f"Roleplay as {character} from {book_name}.", f"Your role is to be {character} from {book_name}."],
            "natural": {
                "character_profile": [f"The profile of {character} is as follows:\n{character_profile}", f"Here is the profile of {character}:\n{character_profile}", f"Your profile is: \n{character_profile}", f"Here is some information about {character}:\n{character_profile}", f"The background of {character} is as follows:\n{character_profile}"],
                "current_scenario": [f"The current scenario is:\n{scenario}", f"Current scenario:\n{scenario}", f"The situation you are in is:\n{scenario}", f"Here is the situation you are in:\n{scenario}"],
                "current_scenario_with_plot_summary": [f"The current scenario and its background are:\nBackground: {background}\nCurrently: {scenario}", f"Current scenario and the background:\nScenario: {scenario}\nMore Background: {background}", f"The situation you are in is:\nStory arc summary: {background}\nCurrent scenario: {scenario}", f"Here is the situation you are in:\nSummary of relevant plots: {background}\nScenario: {scenario}"],
                "other_characters_profile": [f"Here is the your knowledge about the other characters:\n{other_character_profiles_str}", f"Information about other characters:\n{other_character_profiles_str}", f"The background of other characters is as follows:\n{other_character_profiles_str}"],
                "thought": [f"Your thoughts are:\n{motivation}", f"Your thoughts in this situation are:\n{motivation}", f"Your inner thoughts are:\n{motivation}", f"Your inner monologue is:\n{motivation}", f"Your inner thoughts in the scenario are:\n{motivation}"],
                "requirements": [output_format, "" if thoughtless else output_format],
            },
            "=": {
                "decorator": ["==={}===", "=={}==", "={}="],
            },
            "#": {
                "decorator": ["#{}", "# {}", "## {}", "### {}"],
            }, 
            "*": {
                "decorator": ["**{}**", "*{}*", "***{}***"],
            },
            "pieces":{
                "character_profile": ["Character Profile", f"The profile of {character}", f"{character}'s profile"],
                "current_scenario": ["Current Scenario", "The situation you are in", "Scenario"],
                "plot_summary": ["Summary of Relevant Plots", "Background", "Story Arc", "Plot Summary"],
                "thought": [f"{character}'s Thought", "Your thoughts", "Your inner thoughts", "Your inner monologue"],
                "other_characters_profile": [f"Information about other characters", f"The background of other characters", f"Other characters' profiles"],
                "requirements": ["Requirements", "Instructions for roleplaying"],
            }
        }

        # Randomly select a style
        current_style = random.choice(styles)
        
        # Start with a random beginning template
        system_prompt = random.choice(templates["begin"]) + "\n\n"
        
        # Add decorated sections based on style
        if current_style == 'natural':
            # Natural style without decorators
            system_prompt += random.choice(templates["natural"]["character_profile"]) + "\n\n"

            if exclude_plot_summary or random.random() < 0.5:
                system_prompt += random.choice(templates["natural"]["current_scenario"]) + "\n\n"
            else:
                # use Plot Summary in 50% cases
                system_prompt += random.choice(templates["natural"]["current_scenario_with_plot_summary"]) + "\n\n"

            if other_character_profiles_str:
                system_prompt += random.choice(templates["natural"]["other_characters_profile"]) + "\n\n"
            
            if motivation:
                system_prompt += random.choice(templates["natural"]["thought"]) + "\n\n"
            
            if add_rag:
                system_prompt += "Relevant Background Information: \n{retrieved_knowledge}\n\n"

            system_prompt += random.choice(templates["natural"]["requirements"]) + "\n\n"
        else:
            # Styled with decorators
            decorator = random.choice(templates[current_style]["decorator"])
            
            # Character profile section
            section_title = random.choice(templates["pieces"]["character_profile"])
            system_prompt += decorator.format(section_title) + "\n"
            system_prompt += character_profile + "\n\n"
            
            if not exclude_plot_summary and random.random() < 0.5:
                # use Plot Summary in 50% cases
                # Plot summary section
                section_title = random.choice(templates["pieces"]["plot_summary"])
                system_prompt += decorator.format(section_title) + "\n"
                system_prompt += background + "\n\n"

            # Current scenario section
            section_title = random.choice(templates["pieces"]["current_scenario"])
            system_prompt += decorator.format(section_title) + "\n"
            system_prompt += f"{scenario}\n\n"

            if other_character_profiles_str:
                section_title = random.choice(templates["pieces"]["other_characters_profile"])
                system_prompt += decorator.format(section_title) + "\n"
                system_prompt += other_character_profiles_str + "\n\n"

            # Thought section (if not empty)
            if motivation:
                section_title = random.choice(templates["pieces"]["thought"])
                system_prompt += decorator.format(section_title) + "\n"
                system_prompt += motivation + "\n\n"
            
            if add_rag:
                section_title = "Relevant Background Information"
                system_prompt += decorator.format(section_title) + "\n"
                system_prompt += "{retrieved_knowledge}" + "\n\n"

            # Requirements section (if not empty)
            requirements = random.choice(templates["natural"]["requirements"])
            if requirements:
                section_title = random.choice(templates["pieces"]["requirements"])
                system_prompt += decorator.format(section_title) + "\n"
                system_prompt += requirements + "\n\n"

    return system_prompt


def get_character_prompt_sft_cognitive(book_name, character, character_profile, background, scenario, motivation, thoughtless=False, other_character_profiles=None, exclude_plot_summary=False, fixed_template=False, add_output_example=False, add_rag=False):

    output_format = "Your output should include speech, action, thought. Use [your thought] for thoughts, which others can't see. Use (your action) for actions, which others can see."

    if other_character_profiles:
        assert isinstance(other_character_profiles, Dict)
        other_character_profiles_str = ''

        decorator = random.choice(['*'*30 + '\n\n', '*'*20 + '\n\n', '\n\n', '\n', ''])
        for other_character, profile in other_character_profiles.items():
            if other_character != character:
                other_character_profiles_str += f"{decorator}{other_character}: {profile}\n\n"
    else:
        other_character_profiles_str = ''
    
    if fixed_template:
        if motivation: motivation = f"===Your Inner Thoughts===\n{motivation}\n\n"
        if other_character_profiles_str: other_character_profiles_str = f"===Information about the other Characters===\n{other_character_profiles_str}\n\n"

        system_prompt = f"You are {character} from {book_name}.\n\n==={character}'s Profile===\n{character_profile}\n\n===Current Scenario===\n{scenario}\n\n{other_character_profiles_str}{motivation}\n\n"
        
        if add_rag:
            system_prompt += "===Relevant Background Information==={retrieved_knowledge}\n\n"
        
        system_prompt += f"===Requirements===\n{output_format}\n\n"
        return system_prompt
      
    else:
        styles = ['natural'] * 40 + ['='] * 30 + ['#'] * 20 + ['*'] * 10
        templates = {
            "begin": [f"You are {character}.", f"Play the role of {character}.", f"Imagine you are {character}.", f"Think, speak, and act like {character}.", f"Step into the shoes of {character}.", f"Immerse yourself in the character of {character}.", f"You are roleplaying as {character}.", f"You will be portraying {character}.", f"Roleplay as {character}.", f"Your role is to be {character}.", f"You are {character} from {book_name}.", f"Play the role of {character} from {book_name}.", f"Imagine you are {character} from {book_name}.", f"Think, speak, and act like {character} from {book_name}.", f"Step into the shoes of {character} from {book_name}.", f"Immerse yourself in the character of {character} from {book_name}.", f"You are roleplaying as {character} from {book_name}.", f"You will be portraying {character} from {book_name}.", f"Roleplay as {character} from {book_name}.", f"Your role is to be {character} from {book_name}."],
            "natural": {
                "character_profile": [f"The profile of {character} is as follows:\n{character_profile}", f"Here is the profile of {character}:\n{character_profile}", f"Your profile is: \n{character_profile}", f"Here is some information about {character}:\n{character_profile}", f"The background of {character} is as follows:\n{character_profile}"],
                "current_scenario": [f"The current scenario is:\n{scenario}", f"Current scenario:\n{scenario}", f"The situation you are in is:\n{scenario}", f"Here is the situation you are in:\n{scenario}"],
                "current_scenario_with_plot_summary": [f"The current scenario and its background are:\nBackground: {background}\nCurrently: {scenario}", f"Current scenario and the background:\nScenario: {scenario}\nMore Background: {background}", f"The situation you are in is:\nStory arc summary: {background}\nCurrent scenario: {scenario}", f"Here is the situation you are in:\nSummary of relevant plots: {background}\nScenario: {scenario}"],
                "other_characters_profile": [f"Here is the your knowledge about the other characters:\n{other_character_profiles_str}", f"Information about other characters:\n{other_character_profiles_str}", f"The background of other characters is as follows:\n{other_character_profiles_str}"],
                "thought": [f"Your thoughts are:\n{motivation}", f"Your thoughts in this situation are:\n{motivation}", f"Your inner thoughts are:\n{motivation}", f"Your inner monologue is:\n{motivation}", f"Your inner thoughts in the scenario are:\n{motivation}"],
                "requirements": [output_format, "" if thoughtless else output_format],
            },
            "=": {
                "decorator": ["==={}===", "=={}==", "={}="],
            },
            "#": {
                "decorator": ["#{}", "# {}", "## {}", "### {}"],
            }, 
            "*": {
                "decorator": ["**{}**", "*{}*", "***{}***"],
            },
            "pieces":{
                "character_profile": ["Character Profile", f"The profile of {character}", f"{character}'s profile"],
                "current_scenario": ["Current Scenario", "The situation you are in", "Scenario"],
                "plot_summary": ["Summary of Relevant Plots", "Background", "Story Arc", "Plot Summary"],
                "thought": [f"{character}'s Thought", "Your thoughts", "Your inner thoughts", "Your inner monologue"],
                "other_characters_profile": [f"Information about other characters", f"The background of other characters", f"Other characters' profiles"],
                "requirements": ["Requirements", "Instructions for roleplaying"],
            }
        }

        # Randomly select a style
        current_style = random.choice(styles)

        # Start with a random beginning template
        system_prompt = random.choice(templates["begin"]) + "\n\n"

        # Add decorated sections based on style
        if current_style == 'natural':
            # Natural style without decorators
            system_prompt += random.choice(templates["natural"]["character_profile"]) + "\n\n"

            if exclude_plot_summary or random.random() < 0.5:
                system_prompt += random.choice(templates["natural"]["current_scenario"]) + "\n\n"
            else:
                # use Plot Summary in 50% cases
                system_prompt += random.choice(templates["natural"]["current_scenario_with_plot_summary"]) + "\n\n"

            if other_character_profiles_str:
                system_prompt += random.choice(templates["natural"]["other_characters_profile"]) + "\n\n"

            if motivation:
                system_prompt += random.choice(templates["natural"]["thought"]) + "\n\n"
            
            if add_rag:
                system_prompt += "Relevant Background Information: \n{retrieved_knowledge}\n\n"

            system_prompt += random.choice(templates["natural"]["requirements"]) + "\n\n"
        else:
            # Styled with decorators
            decorator = random.choice(templates[current_style]["decorator"])
            
            # Character profile section
            section_title = random.choice(templates["pieces"]["character_profile"])
            system_prompt += decorator.format(section_title) + "\n"
            system_prompt += character_profile + "\n\n"
            
            if not exclude_plot_summary and random.random() < 0.5:
                # use Plot Summary in 50% cases
                # Plot summary section
                section_title = random.choice(templates["pieces"]["plot_summary"])
                system_prompt += decorator.format(section_title) + "\n"
                system_prompt += background + "\n\n"

            # Current scenario section
            section_title = random.choice(templates["pieces"]["current_scenario"])
            system_prompt += decorator.format(section_title) + "\n"
            system_prompt += f"{scenario}\n\n"

            if other_character_profiles_str:
                section_title = random.choice(templates["pieces"]["other_characters_profile"])
                system_prompt += decorator.format(section_title) + "\n"
                system_prompt += other_character_profiles_str + "\n\n"

            # Thought section (if not empty)
            if motivation:
                section_title = random.choice(templates["pieces"]["thought"])
                system_prompt += decorator.format(section_title) + "\n"
                system_prompt += motivation + "\n\n"
            
            if add_rag:
                section_title = "Relevant Background Information"
                system_prompt += decorator.format(section_title) + "\n"
                system_prompt += "{retrieved_knowledge}" + "\n\n"

            # Requirements section (if not empty)
            requirements = random.choice(templates["natural"]["requirements"])
            if requirements:
                section_title = random.choice(templates["pieces"]["requirements"])
                system_prompt += decorator.format(section_title) + "\n"
                system_prompt += requirements + "\n\n"
        return system_prompt


def get_character_prompt(book_name, character, character_profile, background, scenario, motivation, thoughtless=False, other_character_profiles=None, exclude_plot_summary=False, fixed_template=False, add_output_example=False, add_rag=False):

    if thoughtless:
      output_format = "Your output should include **speech** and **action**. Use (your action) for actions, which others can see."
    else:
      output_format = "Your output should include **thought**, **speech**, and **action**. Use [your thought] for thoughts, which others can't see. Use (your action) for actions, which others can see."

      if add_output_example:
        output_format = f"""
Your output should include **thought**, **speech**, and **action**.
Use [your thought] for thoughts, which others can't see, such as [I'm terrified, but I must appear strong.]. 
Use no tag for your speech, which others can see. 
Use (your action) for actions, which others can see, such as (watches silently, trying to control her fear and anger).
""".strip()

    if other_character_profiles:
        assert isinstance(other_character_profiles, Dict)
        other_character_profiles_str = ''

        decorator = random.choice(['*'*30 + '\n\n', '*'*20 + '\n\n', '\n\n', '\n', ''])
        for other_character, profile in other_character_profiles.items():
            if other_character != character:
                other_character_profiles_str += f"{decorator}{other_character}: {profile}\n\n"
    else:
        other_character_profiles_str = ''
    
    if fixed_template:
        if motivation: motivation = f"===Your Inner Thoughts===\n{motivation}\n\n"
        if other_character_profiles_str: other_character_profiles_str = f"===Information about the other Characters===\n{other_character_profiles_str}\n\n"

        system_prompt = f"You are {character} from {book_name}.\n\n==={character}'s Profile===\n{character_profile}\n\n===Current Scenario===\n{scenario}\n\n{other_character_profiles_str}{motivation}\n\n"
        
        if add_rag:
            system_prompt += "===Relevant Background Information==={retrieved_knowledge}\n\n"
        
        system_prompt += f"===Requirements===\n{output_format}\n\n"
        return system_prompt
      
    else:
        styles = ['natural'] * 40 + ['='] * 30 + ['#'] * 20 + ['*'] * 10
        templates = {
            "begin": [f"You are {character}.", f"Play the role of {character}.", f"Imagine you are {character}.", f"Think, speak, and act like {character}.", f"Step into the shoes of {character}.", f"Immerse yourself in the character of {character}.", f"You are roleplaying as {character}.", f"You will be portraying {character}.", f"Roleplay as {character}.", f"Your role is to be {character}.", f"You are {character} from {book_name}.", f"Play the role of {character} from {book_name}.", f"Imagine you are {character} from {book_name}.", f"Think, speak, and act like {character} from {book_name}.", f"Step into the shoes of {character} from {book_name}.", f"Immerse yourself in the character of {character} from {book_name}.", f"You are roleplaying as {character} from {book_name}.", f"You will be portraying {character} from {book_name}.", f"Roleplay as {character} from {book_name}.", f"Your role is to be {character} from {book_name}."],
            "natural": {
                "character_profile": [f"The profile of {character} is as follows:\n{character_profile}", f"Here is the profile of {character}:\n{character_profile}", f"Your profile is: \n{character_profile}", f"Here is some information about {character}:\n{character_profile}", f"The background of {character} is as follows:\n{character_profile}"],
                "current_scenario": [f"The current scenario is:\n{scenario}", f"Current scenario:\n{scenario}", f"The situation you are in is:\n{scenario}", f"Here is the situation you are in:\n{scenario}"],
                "current_scenario_with_plot_summary": [f"The current scenario and its background are:\nBackground: {background}\nCurrently: {scenario}", f"Current scenario and the background:\nScenario: {scenario}\nMore Background: {background}", f"The situation you are in is:\nStory arc summary: {background}\nCurrent scenario: {scenario}", f"Here is the situation you are in:\nSummary of relevant plots: {background}\nScenario: {scenario}"],
                "other_characters_profile": [f"Here is the your knowledge about the other characters:\n{other_character_profiles_str}", f"Information about other characters:\n{other_character_profiles_str}", f"The background of other characters is as follows:\n{other_character_profiles_str}"],
                "thought": [f"Your thoughts are:\n{motivation}", f"Your thoughts in this situation are:\n{motivation}", f"Your inner thoughts are:\n{motivation}", f"Your inner monologue is:\n{motivation}", f"Your inner thoughts in the scenario are:\n{motivation}"],
                "requirements": [output_format, "" if thoughtless else output_format],
            },
            "=": {
                "decorator": ["==={}===", "=={}==", "={}="],
            },
            "#": {
                "decorator": ["#{}", "# {}", "## {}", "### {}"],
            }, 
            "*": {
                "decorator": ["**{}**", "*{}*", "***{}***"],
            },
            "pieces":{
                "character_profile": ["Character Profile", f"The profile of {character}", f"{character}'s profile"],
                "current_scenario": ["Current Scenario", "The situation you are in", "Scenario"],
                "plot_summary": ["Summary of Relevant Plots", "Background", "Story Arc", "Plot Summary"],
                "thought": [f"{character}'s Thought", "Your thoughts", "Your inner thoughts", "Your inner monologue"],
                "other_characters_profile": [f"Information about other characters", f"The background of other characters", f"Other characters' profiles"],
                "requirements": ["Requirements", "Instructions for roleplaying"],
            }
        }

        # Randomly select a style
        current_style = random.choice(styles)

        # Start with a random beginning template
        system_prompt = random.choice(templates["begin"]) + "\n\n"

        # Add decorated sections based on style
        if current_style == 'natural':
            # Natural style without decorators
            system_prompt += random.choice(templates["natural"]["character_profile"]) + "\n\n"

            if exclude_plot_summary or random.random() < 0.5:
                system_prompt += random.choice(templates["natural"]["current_scenario"]) + "\n\n"
            else:
                # use Plot Summary in 50% cases
                system_prompt += random.choice(templates["natural"]["current_scenario_with_plot_summary"]) + "\n\n"

            if other_character_profiles_str:
                system_prompt += random.choice(templates["natural"]["other_characters_profile"]) + "\n\n"

            if motivation:
                system_prompt += random.choice(templates["natural"]["thought"]) + "\n\n"
            
            if add_rag:
                system_prompt += "Relevant Background Information: \n{retrieved_knowledge}\n\n"

            system_prompt += random.choice(templates["natural"]["requirements"]) + "\n\n"
        else:
            # Styled with decorators
            decorator = random.choice(templates[current_style]["decorator"])
            
            # Character profile section
            section_title = random.choice(templates["pieces"]["character_profile"])
            system_prompt += decorator.format(section_title) + "\n"
            system_prompt += character_profile + "\n\n"
            
            if not exclude_plot_summary and random.random() < 0.5:
                # use Plot Summary in 50% cases
                # Plot summary section
                section_title = random.choice(templates["pieces"]["plot_summary"])
                system_prompt += decorator.format(section_title) + "\n"
                system_prompt += background + "\n\n"

            # Current scenario section
            section_title = random.choice(templates["pieces"]["current_scenario"])
            system_prompt += decorator.format(section_title) + "\n"
            system_prompt += f"{scenario}\n\n"

            if other_character_profiles_str:
                section_title = random.choice(templates["pieces"]["other_characters_profile"])
                system_prompt += decorator.format(section_title) + "\n"
                system_prompt += other_character_profiles_str + "\n\n"

            # Thought section (if not empty)
            if motivation:
                section_title = random.choice(templates["pieces"]["thought"])
                system_prompt += decorator.format(section_title) + "\n"
                system_prompt += motivation + "\n\n"
            
            if add_rag:
                section_title = "Relevant Background Information"
                system_prompt += decorator.format(section_title) + "\n"
                system_prompt += "{retrieved_knowledge}" + "\n\n"

            # Requirements section (if not empty)
            requirements = random.choice(templates["natural"]["requirements"])
            if requirements:
                section_title = random.choice(templates["pieces"]["requirements"])
                system_prompt += decorator.format(section_title) + "\n"
                system_prompt += requirements + "\n\n"
        return system_prompt


def get_environment_prompt(major_characters, scenario):
	ENVIRONMENT = "Environment"
	major_characters = [c for c in major_characters if c != ENVIRONMENT]

	model_roles = [
		"an environment model",
		"a world model",
		"a world simulator",
		"an environment simulator"
	]

	prompt = f"""You are {random.choice(model_roles)} for a role-playing game. Your task is to provide the environmental feedback: Based on the characters' interactions, dialogues, and actions, describe the resulting changes in the environment. This includes:
   - Physical changes in the setting
   - Reactions of background characters or crowds
   - Ambient sounds, weather changes, or atmospheric shifts
   - Any other relevant environmental details

Your descriptions should be vivid and help set the scene, but avoid dictating the actions or dialogue of the main characters (including {major_characters}).

Important notes:
- You may include actions and reactions of minor characters or crowds, as long as they're not main characters (including {major_characters}).
- Keep your environmental descriptions concise but impactful, typically 1-3 sentences.
- Respond to subtle cues in the characters' interactions to create a dynamic, reactive environment.
- Your output should match the tone, setting, and cultural context of the scenario.

===The scenario is as follows===
{scenario}"""

	return prompt


def get_nsp_prompt(all_characters, scenario):
    ENVIRONMENT = "Environment"
    prompt = f"""
Your task is to predict the next speaker for a role-playing game. That is, you need to determine which character (or the {ENVIRONMENT}) might act next based on their previous interactions. The {ENVIRONMENT} is a special role that provides the environmental feedback. Choose a name from this list: {all_characters}. If it's unclear who should act next, output "random". If you believe the scene or conversation should conclude, output "<END CHAT>".

===The scenario is as follows===
{scenario}""".strip()
	
    return prompt


from typing import Dict

def print_conversation_to_file(conversation_data: Dict, file_path: str):
	"""
	Write the scenario, actor prompt, user prompt, and the formatted conversation to a file.
	:param conversation_data: The dictionary containing scene details, actor prompt, user prompt, and conversation entries.
	:param file_path: The path to the file where the output will be written.
	"""
	# Extract components from the conversation data
	scene = conversation_data['scene']
	actor_prompt = conversation_data.get("actor_prompt", "N/A")
	user_prompt = conversation_data.get("user_prompt", "N/A")
	conversation = conversation_data["conversation"]

	with open(file_path, 'a', encoding='utf-8') as file:
		file.write("\n=== Scene Description ===\n")
		file.write(f"Scenario: {scene['scenario']}\n")
		
		file.write("\n=== Actor Prompt ===\n")
		file.write(f"{actor_prompt}\n")
		
		file.write("\n=== User Prompt ===\n")
		file.write(f"{user_prompt}\n")
		
		file.write("\n=== Conversation ===\n")
		for turn in conversation:
			from_ = turn["from"]
			file.write(f"\n=== {from_} ===\n")
			message = turn["message"]
			file.write(f"{message}\n\n")

	return 

def extract_json(text, **kwargs):
	def _fix_json(json_response):
		prompt = f'''I will provide you with a JSON string that contains errors, making it unparseable by `json.loads`. The most common issue is the presence of unescaped double quotes inside strings. Your task is to output the corrected JSON string. The JSON string to be corrected is:
{json_response}
'''

		response = get_response(model=kwargs['model'], messages=[{"role": "user", "content": prompt}])

		logger.info(f'fixed json: {response}')	

		return response

	def _extract_json(text):
		# Use regular expressions to find all content within curly braces
		orig_text = text

		text = re.sub(r'"([^"\\]*(\\.[^"\\]*)*)"', lambda m: m.group().replace('\n', r'\\n'), text) 
		
		#json_objects = re.findall(r'(\{[^{}]*\}|\[[^\[\]]*\])', text, re.DOTALL)

		def parse_json_safely(text):
			try:
				result = json.loads(text)
				return result
			except json.JSONDecodeError:
				results = []
				start = 0
				while start < len(text):
					try:
						obj, end = json.JSONDecoder().raw_decode(text[start:])
						results.append(obj)
						start += end
					except json.JSONDecodeError:
						start += 1
				
				if results:
					longest_json = max(results, key=lambda x: len(json.dumps(x)))
					return longest_json
				else:
					return None
		
		extracted_json = parse_json_safely(text)
		if extracted_json:
			return extracted_json
		else:
			logger.error('Error parsing response: ', orig_text)
			return ""

	res = _extract_json(text)

	if res:
		return res
	else:
		return _extract_json(_fix_json(text))

def ensure_scenes(cand_scenes, **kwargs):
	if isinstance(cand_scenes, list) and len(cand_scenes) > 0 and {"scenario", "actor_role", "user_role", "topic", "leader", "max_rounds"}.issubset(cand_scenes[0].keys()):
		return cand_scenes
	else:
		return False

def conversation_to_str(conversation, background={}, to_remove_inner_thoughts=True):
	conversation_text = ''

	for b_k, b_v in background.items():
		conversation_text += '{}:\n'.format(b_k) + b_v + '\n\n'

	conversation_text += 'Conversation:\n'
	for message in conversation:
		c = message['character']
		if 'message' in message: 
			m = message['message']
		else:
			m = message['dialogues']
		if remove_inner_thoughts:
			m = remove_inner_thoughts(m)
	
		if not m.startswith(c):
			m = c + ': ' + m
		conversation_text += m + '\n\n'

	return conversation_text

def get_response_with_retry(**kwargs):

	return get_response_json([], **kwargs)

def get_response_json(post_processing_funcs=[extract_json], **kwargs):
	"""
    Get and process a response from an LLM with retries and error handling.
    
    This function handles:
    1. Getting responses from the LLM with retries
    2. Processing responses through a pipeline of post-processing functions
    3. Fallback handling for parsing failures
    
    Args:
        post_processing_funcs (list): List of functions to process the LLM response, defaults to [extract_json]
        **kwargs: Additional arguments passed to get_response(), including:
            - messages: List of message dicts for the LLM
            - model: Name of LLM model to use
            - max_retry: Max number of retry attempts (default 5)
            
    Returns:
        dict: Processed JSON response from the LLM, or error dict if parsing fails
    """
	nth_generation = 0
	json_response = ""
    
	while (True):
		response = get_response(**kwargs, nth_generation=nth_generation)
        # 如果 strict_level > 3 则返回的 response = ""
		if response is None:
			nth_generation += 1
			time.sleep(0.5)
			if nth_generation > kwargs.get('max_retry', 10):
				response = ""
				break
			else:
				continue
               
        # 实际运行时，传入的 post_processing_funcs 为空, 因此 json response = response = "", if strict_level > 3
		for post_processing_func in post_processing_funcs:
			response = post_processing_func(response, **kwargs)
		json_response = response 
		
		if json_response:
			break
		else:
			nth_generation += 1
			time.sleep(1.0)
			print(f'Error parsing response, retrying... {nth_generation}th time')
			if nth_generation > kwargs.get('max_retry', 20):
				json_response = ""
				break
	
	return json_response

def print_json(data):
	print(json.dumps(data, ensure_ascii=False, indent=2))

def save_json(data: List[Dict], file_path: str):
	with open(file_path, "w", encoding='utf-8') as f:
		json.dump(data, f, ensure_ascii=False, indent=2)

def read_json(file_path: str) -> List[Dict]:
	with open(file_path, 'r', encoding='utf-8') as f:
		data = json.load(f)
	return data

def tokenize_words(text):
	import regex
	pattern = r'\b\w+\b|[\u4e00-\u9fff]|[\u3040-\u309F\u30A0-\u30FF]|\d|[\p{P}\p{S}]'
	tokens = regex.findall(pattern, text)


	tokens_expanded = []
	for token in tokens:
		if re.match(r'[\u4e00-\u9fff]|[\u3040-\u309F\u30A0-\u30FF]', token):
			tokens_expanded.extend(list(token))
		else:
			tokens_expanded.append(token)
	return tokens_expanded

def fix_repeation(response):
	"""
	Fix repetitive text patterns in the response by detecting and removing repetitions.
	
	This function handles three types of repetition detection:
	1. Long letter substrings (100+ characters)
	2. Consecutive repetitions of token sequences
	3. Non-consecutive repetitions of token sequences
	
	Args:
		response (str): The text response to check for repetitions
		
	Returns:
		str: The fixed text with repetitions removed if repetitions were found
		False: If no repetitions were detected
	"""

	def detect_repetitions(tokens, min_length=5, max_length=30, threshold=0.1):
		"""Check for consecutive repetitions of token sequences"""
		total_length = len(tokens)
		repetitions = 0
	
		# Try different lengths of subsequences
		for length in range(min_length, min(max_length + 1, total_length + 1)):
			for i in range(total_length - length + 1):
				substr = tokens[i:i + length] 

				# Check if this subsequence repeats consecutively up to 4 times
				is_repeated = True
				for repeat_idx in range(1, 5):
					check_pos = i + (repeat_idx * length)

					if tokens[check_pos:check_pos + length] != substr:
						is_repeated = False
						break
				
				if is_repeated:
					return tokens[:i + length]  # Return text up to first repetition

		return False

	def detect_repetitions2(tokens, min_length=15, max_length=30, threshold=0.1):
		"""Check for non-consecutive repetitions of token sequences"""
		total_length = len(tokens)
		repetitions = 0
		
		first_repeat_idx = 999999999999999
		first_start_idx = {}

		# Try different lengths of subsequences
		for length in range(min_length, min(max_length + 1, total_length + 1)):
			substr_count = {}

			for i in range(total_length - length + 1):
				substr = tuple(tokens[i:i + length]) 
				if substr_count.get(substr, 0) > 0:
					# Found a repeat - check if it's far enough from first occurrence
					if i - first_start_idx[substr] >= length:			
						first_repeat_idx = min(first_repeat_idx, i)
				else:
					first_start_idx[substr] = i

				substr_count[substr] = substr_count.get(substr, 0) + 1
			
			repetitions += sum(count > 1 for count in substr_count.values())

		repetition_rate = repetitions / total_length if total_length else 0
	
		if first_repeat_idx < 999999999999999:
			return tokens[:first_repeat_idx]  # Return text up to first repetition
		else:
			return False

	def concatenate_tokens(tokens):
		"""Reconstruct text from tokens with proper spacing and punctuation"""
		text = ""
		last_type = None
		
		for token in tokens:
			# Determine token type (CJK, punctuation, or other)
			current_type = 'CJK' if re.match(r'[\u4e00-\u9fff]|[\u3040-\u309F\u30A0-\u30FF]', token) else 'Other'
			import string
			if token in string.punctuation:
				current_type = 'P'

			# Add space between certain token types
			if last_type in ['Other', 'P'] and current_type == 'Other':
				text += " " + token
			else:
				text += token

			last_type = current_type
		
		# Add appropriate ending punctuation based on last character
		if re.match(r'[a-zA-Z0-9]+$', text[-1]):
			text += '.'
		if re.match(r'[\u4e00-\u9fff]+$', text[-1]):
			text += '。'
		if re.match(r'[\u3040-\u309F\u30A0-\u30FF]+$', text[-1]):
			text += '。'

		return text

	def find_long_letter_substrings(s):
		"""Find substrings of letters that are 100+ characters long"""
		pattern = r'[a-zA-Z]{100,}'
		matches = re.findall(pattern, s)
		return matches

	repeat_sign = False 

	# First check for very long letter sequences
	_ = find_long_letter_substrings(response)
	if _:
		for substr in _:
			response = response.replace(substr, substr[:20])
		repeat_sign = True

	# Then check for token sequence repetitions
	tokens = tokenize_words(response)
	_ = detect_repetitions(tokens)
	if _ == False:  # If no consecutive repetitions found
		_ = detect_repetitions2(tokens)  # Check for non-consecutive repetitions

	if _:
		response = concatenate_tokens(_)
		repeat_sign = True

	if repeat_sign:
		return response  # Return fixed text if repetitions were found
	else:
		return False  # Return False if no repetitions detected

from collections import Counter
import math

def avg(list):
	return sum(list) / len(list)

def entropy(text):
	words = tokenize_words(text)
	counter = Counter(words)
	total = sum(counter.values())
	probs = [count/total for count in counter.values()]
	
	return -sum(p * math.log2(p) for p in probs)

def ttr(text):
	words = tokenize_words(text)
	return len(set(words)) / len(words)


from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from rouge import Rouge
import nltk

def calculate_bleu_rouge(reference, simulation):

    simulation_str = '\n\n'.join([m['content'].strip('\n') for m in simulation])
    reference_str = '\n\n'.join([f"{m['character']}: {m['message']}".strip('\n') for m in reference])

    # remove the speaker name
    reference_tokens = word_tokenize(reference_str.lower())
    simulation_tokens = word_tokenize(simulation_str.lower())
    bleu = sentence_bleu([reference_tokens], simulation_tokens)
    rouge_l = Rouge().get_scores(simulation_str, reference_str)[0]['rouge-l']['f']
    return bleu, rouge_l


if __name__ == '__main__':
	messages = [{"role": "system", "content": "Hello, how are you?"}]
	model = "gpt-4o"

	print(get_response(model, messages))
		
