import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
import random

from gca_evaluation.utils import (
    get_character_prompt_reasoning_w_fix_cognitive,
    get_character_prompt_reasoning,
    get_character_prompt,
    get_character_prompt_reasoning_w_free_cognitive,
    get_character_prompt_reasoning_w_fix_cognitive_auto,
    remove_inner_thoughts, load_json,
    get_character_prompt_sft_cognitive
)


def get_prompt_fucntion(method: str):
    if method == 'fixed':
        return get_character_prompt_reasoning_w_fix_cognitive
    elif method == 'free':
        return get_character_prompt_reasoning_w_free_cognitive
    elif method == 'reasoning':
        return get_character_prompt_reasoning
    elif method == "auto_fixed":
        return get_character_prompt_reasoning_w_fix_cognitive_auto
    elif method == "sft_cognitive":
        return get_character_prompt_sft_cognitive
    else:
        return get_character_prompt



def load_example_messages(method='fixed', pre_nums=0, use_random=False, seed=42):
    random.seed(seed)
    ENVIRONMENT = 'Environment'
    NSP = "NSP"
    example_fp = "/Users/liucheng/Desktop/RL/role_rl/CoSER/data/test/test_set_random40.json"
    examples = load_json(example_fp)
    if use_random:
        example = random.choice(examples)
    else:
        example = examples[0]

    # 加载测试字段
    book_title = example['book']
    plot = example['plot']
    conversation = example
    character_profiles = example['character_profiles']
    plot_characters = [ c['name'] for c in plot['key_characters']] 
    speaking_characters_w_env = conversation['speaking_characters_w_env']

    if ENVIRONMENT not in speaking_characters_w_env:
        speaking_characters_w_env.append(ENVIRONMENT)
    involved_character_profiles = {}

    for character in speaking_characters_w_env:    
        if character == ENVIRONMENT:
            continue
        
        character_profile = character_profiles.get(character, '')
        if character in plot_characters:
            character_info = [c for c in plot['key_characters'] if c.get('name', '') == character][0]
            if 'description' in character_info:
                character_profile = character_info.get('description', '').strip('\n') + '\n\n' + character_profile.strip('\n')
                
        character_profile = character_profile.strip(' \n')
        if character_profile != '':
            involved_character_profiles[character] = character_profile

    for character in speaking_characters_w_env + [NSP]:
        if character == NSP or character == ENVIRONMENT:
            continue
        else:
            character_profile = involved_character_profiles.get(character, '')
            if character in plot_characters:
                character_info = [c for c in plot['key_characters'] if c.get('name', '') == character][0]
            character_profile = character_profile.strip(' \n')
            find_motivation = [ c.get('motivation', '') for c in conversation['key_characters'] if c.get('name', '') == character]
            motivation = find_motivation[0] if find_motivation else ''
            break

    get_prompt = get_prompt_fucntion(method)

    system_prompt = get_prompt(
        book_title, character, character_profile, plot["summary"],
        conversation["scenario"], motivation, thoughtless=False,
        other_character_profiles=involved_character_profiles,
        exclude_plot_summary=True, fixed_template=True,
        add_output_example=True, add_rag=False
    )

    # system_prompt += "\n\nSpeak concisely as humans, instead of being verbose. Limit your response to 60 words.\n\n"
    
    dialogues = []
    dialogue_str = "===Conversation Start===\n\n"
    for dialogue in conversation['dialogues'][:pre_nums]:
        message = dialogue['message']
        current_character = dialogue['character']
        if current_character == character:
            dialogues.append({"role": "assistant", "content": f"{character}: {message}"})
            dialogue_str += f"{character}: {message}\n"
        else:
            message = remove_inner_thoughts(message)
            dialogues.append({"role": "user", "content": f"{current_character}: {message}"})
            dialogue_str += f"{current_character}: {message}\n"

    reference = conversation['dialogues'][0]["character"] + ": " + conversation['dialogues'][0]["message"]

    messages = [
        {"role": "system", "content": system_prompt, "reference": reference},
        # {"role": "user", "content": dialogue_str}
    ] + dialogues

    return messages