import json
from math import e 
from utils import get_response
import argparse
from tqdm import tqdm
from utils import setup_logger
from agent import Agent
import random
import os
import time
from utils import get_environment_prompt, get_nsp_prompt, get_character_prompt
from utils import get_response_json, extract_json, parse_reasoning_response
from utils import remove_inner_thoughts, calculate_bleu_rouge
from utils import (
    generate_cognitive_prompt_stage1, 
    get_character_prompt_w_cognitive_stage2,
    get_character_prompt_reasoning_w_fix_cognitive,
    get_character_prompt_reasoning_w_free_cognitive,
    get_character_prompt_reasoning,
    # get_character_prompt_reasoning_w_fix_cognitive_wo_self,
    get_character_prompt_reasoning_w_fix_cognitive_updates,
    get_character_prompt_sft_cognitive
)

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))
from model.chatgpt import get_chatgpt_response

random.seed(42)

logger = None

# Set up command line argument parser
parser = argparse.ArgumentParser(
    description='Evaluate role-playing language models via given-circumstance acting (GCA)'
)

# Input/output paths
parser.add_argument(
    '--use_reasoning',
    action='store_true',
    default=False,
    help='Whether to use reasoning prompt template in the model'
)

parser.add_argument(
    '--cognitive_method',
    type=str,
    default='none',
    choices=['free', 'fixed', "two_stage", 'none', 'only_other', 'fixed_with_updates', 'sft_cognitive'],
    help='Cognitive method to use in the model'
)

parser.add_argument(
    '--test_file',
    type=str,
    default='data/test/test_set.json',
    help='Path to the test dataset'
)
parser.add_argument(
    '--book_data',
    type=str,
    default='data/final',
    help='Path to the folder containing complete curated data of each book, used when retrieval augmentation is enabled.'
)

# Model configuration
parser.add_argument(
    '--actor_model',
    type=str,
    default='gpt-4o',
    help='Name of the model to use for role-playing'
)
parser.add_argument(
    '--judge_model',
    type=str,
    default='gpt-4o',
    help='Name of the model to use for LLM judging'
)
parser.add_argument(
    '--env_model',
    type=str,
    default='gpt-4o',
    help='Name of the model to use for environment response'
)
parser.add_argument(
    '--nsp_model',
    type=str,
    default='gpt-4o-mini',
    help='Name of the model to use for next-speaker prediction, default to gpt-4o-mini, but recommend Coser-70B or self-deployed models for better cost-efficiency.'
)

# Runtime settings
parser.add_argument(
    '--continue_from',
    type=int,
    default=0,
    help='Start GCA from the i-th round. The previous rounds will use the ground truth conversations.'
)
parser.add_argument(
    '--wo_thought',
    default=False,
    action='store_true',
    help='Disable inner thoughts in generation'
)
parser.add_argument(
    '--retrieval',
    type=str,
    default=None,
    choices=[None, 'raw_text', 'expr1', 'expr3', 'conv1', 'expr3_conv1', 'expr10_conv1'],
    help='Target for retrieval'
)
parser.add_argument(
    '--regenerate',
    action='store_true',
    help='Regenerate the simulation results'
)
parser.add_argument(
    '--reevaluate',
    action='store_true',
    help='Reevaluate the simulation results'
)
parser.add_argument(
    '--nth_exp',
    type=int,
    default=0,
    help='Experiment ID. Results will be reused for same ID. Set to -1 to run 3 experiments.'
)
parser.add_argument(
    '--num_workers',
    type=int,
    default=1,
    help='Number of parallel workers (default: 1)'
)

# Parse arguments
args = parser.parse_args()

ENVIRONMENT = 'Environment'
NSP = "NSP"
special_characters = [NSP, ENVIRONMENT]


def get_current_time_str():
    """
    Returns the current time as a string formatted as MMDD-HHMM. like 0407_1130
    """
    return time.strftime("%m%d_%H%M", time.localtime())


def get_method_suffix():

    time_suffix = f"-{get_current_time_str()}"
    # time_suffix = "0514_1248"
    suffix = ""

    if not args.use_reasoning:
        suffix = "vanilla"
    
    else:
        if args.cognitive_method == 'fixed':
            suffix = "cognitive_fixed"
        elif args.cognitive_method == 'free':
            suffix = "cognitive_free"
        elif args.cognitive_method == 'two_stage':
            suffix = "cognitive_two_stage"
        elif args.cognitive_method == 'none':
            suffix = "reasoning"
        elif args.cognitive_method == 'only_other':
            suffix = "cognitive_fixed_only_other"
        elif args.cognitive_method == 'fixed_with_updates':
            suffix = "cognitive_fixed_with_updates"
        elif args.cognitive_method == "sft_cognitive":
            suffix = "sft_cognitive"
        else:
            raise ValueError(f"Invalid cognitive method: {args.cognitive_method}")
        
    return f"{suffix}{time_suffix}"
    
        
suffix = get_method_suffix()


def test_safe(test_str):
    response = get_chatgpt_response(
        model = "gpt-4o-nlp",
        messages = [{"role":"user", "content": f"{test_str}, please answer yes"}]
    )
    if not response:
        return False
    else:
        return True


def gca_simulation(test_file, actor_model, env_model, nsp_model, retrieval, nth_exp=0):
    """
    Conducts Given-Circumstance Acting (GCA) simulation where LLM agents role-play characters in specific scenarios.
    The simulation involves multiple agents:
    - Character agents (using actor_model) that role-play the characters
    - Environment agent (using env_model) that takes a special role of "Environment" and provides environmental feedback
    - Next-Speaker Predictor (using nsp_model) that determines the speaking agent in each round

    Each character agent is initialized with relevant character data. 
    The agents then engage in multi-turn dialogue, with the NSP model directing speaker transitions.

    Args:
        test_file (str): Path to JSON file containing test cases.
        actor_model (str): Model name for character role-playing agents
        env_model (str): Model name for environment agent
        nsp_model (str): Model name for next speaker prediction
        retrieval (str, optional): Type of retrieval data to enhance role-playing. Defaults to None (no retrieval).
        nth_exp (int, optional): Experiment ID. 

    Returns:
        list: Simulation results for each scenario.
    """

    # Set up caching file for model outputs
    from utils import set_cache_path
    cache_path = f'.cache/{actor_model+"_"+suffix}.pkl'
    if nth_exp > 0:
        cache_path = f'{cache_path}-repeat={nth_exp}'
    set_cache_path(cache_path)
    
    # Load test set
    test_dataset = json.load(open(test_file, 'r'))
    results = []

    # Configure output path based on model and retrieval settings
    actor_setting = f'{actor_model}{"_" + suffix}{"_rag=" + retrieval if retrieval else ""}'
    simulation_path = f'exp/simulation/{test_file.split("/")[-1].replace(".json", "")}_{actor_setting}.json'

    logger.info(f'Conducting GCA Simulation for {actor_setting} on {test_file}\n\nThe results will be saved to {simulation_path}')

    # Return cached results if available 
    if os.path.exists(simulation_path) and not args.regenerate:
        return json.load(open(simulation_path, 'r'))

    # Traverse each test sample in the test dataset
    for circumstance in test_dataset:
        # collect scenario metadata and context
        book_title = circumstance['book']
        plot = circumstance['plot']
        i_p = plot['i_p'] 
        conversation = circumstance
        i_c = conversation['i_c']
        character_profiles = circumstance['character_profiles']

        logger.info(f'==========Book {book_title}==========')

        # Load additional book data if retrieval is enabled
        if retrieval:
            book_database = json.load(open(f'{args.book_data}/{book_title}.json', 'r'))

        # Identify the character lists
        plot_characters = [ c['name'] for c in plot['key_characters']] 
        speaking_characters_w_env = conversation['speaking_characters_w_env']
        if ENVIRONMENT not in speaking_characters_w_env:
            speaking_characters_w_env.append(ENVIRONMENT)
        major_characters = conversation['major_characters']

        character_agents = {}
        involved_character_profiles = {}

        # 新增： 存储每个角色认知状态
        character_cognitive_states = {character: None for character in speaking_characters_w_env}

        # Build enhanced character profiles combining scenario and plot information
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

        # Create agents for all roles (characters + NSP)
        for character in speaking_characters_w_env + [NSP]:    
            # Configure agent based on role type
            if character == NSP:
                # Next Speaker Predictor agent
                system_prompt = get_nsp_prompt(speaking_characters_w_env, conversation['scenario'])
                character_database = None
            elif character == ENVIRONMENT:
                # Environment description agent
                system_prompt = get_environment_prompt(major_characters, conversation['scenario'])
                character_database = None
            else:
                # Character role-playing agent
                if retrieval and character in book_database['character_datasets']:
                    # Set up retrieval database for character context
                    character_database = book_database['character_datasets'][character]
                    involved_plots = [_['i_p'] for _ in character_database['plots']] + \
                                   [_['i_p'] for _ in character_database['conversations']] + \
                                   [_['i_p'] for _ in character_database['utterances']]
                    involved_plots = sorted(set(involved_plots))
                    character_database['detailed_plots'] = [ book_database['plots'][i] for i in involved_plots ] 
                else:
                    character_database = None

                # Build character context from profile and plot
                character_profile = involved_character_profiles.get(character, '')
                if character in plot_characters:
                    character_info = [c for c in plot['key_characters'] if c.get('name', '') == character][0]
                character_profile = character_profile.strip(' \n')

                # Get character motivation if specified
                find_motivation = [ c.get('motivation', '') for c in conversation['key_characters'] if c.get('name', '') == character]
                motivation = find_motivation[0] if find_motivation else ''

                # Configure prompt based on model type
                # 默认需要添加一个例子以便于能够理解 thought 和 action 的输出
                add_output_example = False if 'coser' in actor_model.lower() else True 

                if args.use_reasoning:
                    # 这个参数记得添加进去
                    if args.cognitive_method == 'none':
                        system_prompt = get_character_prompt_reasoning(
                            book_title, character, character_profile, plot["summary"],
                            conversation["scenario"], motivation, thoughtless=args.wo_thought,
                            other_character_profiles=involved_character_profiles,
                            exclude_plot_summary=True, fixed_template=True,
                            add_output_example=add_output_example, add_rag=retrieval
                        )

                    elif args.cognitive_method == 'two_stage':
                        cognitive_system_prompt = generate_cognitive_prompt_stage1(
                            book_title, character, character_profile, plot["summary"],
                            conversation["scenario"], motivation,
                            other_character_profiles=involved_character_profiles,
                            exclude_plot_summary=True, fixed_template=True,
                            add_output_example=add_output_example, add_rag=retrieval
                        )
                        system_prompt_params = {
                            'book_name': book_title,
                            'character': character,
                            'character_profile': character_profile,
                            'background': plot["summary"],
                            'scenario': conversation["scenario"],
                            'motivation': motivation,
                            'other_character_profiles': involved_character_profiles,
                            'cognitive_profile': None,
                            "function_kwargs": {
                                'exclude_plot_summary': True,
                                'fixed_template': True,
                                'add_output_example': add_output_example,
                                'add_rag': retrieval
                            }
                        }
                        system_prompt = ""

                    elif args.cognitive_method == 'fixed':
                        system_prompt = get_character_prompt_reasoning_w_fix_cognitive(
                            book_title, character, character_profile, plot["summary"],
                            conversation["scenario"], motivation, thoughtless=args.wo_thought,
                            other_character_profiles=involved_character_profiles,
                            exclude_plot_summary=True, fixed_template=True,
                            add_output_example=add_output_example, add_rag=retrieval
                        )
                    elif args.cognitive_method == 'free':
                        system_prompt = get_character_prompt_reasoning_w_free_cognitive(
                            book_title, character, character_profile, plot["summary"],
                            conversation["scenario"], motivation, thoughtless=args.wo_thought,
                            other_character_profiles=involved_character_profiles,
                            exclude_plot_summary=True, fixed_template=True,
                            add_output_example=add_output_example, add_rag=retrieval
                        )

                    # elif args.cognitive_method == 'only_other':
                    #     system_prompt = get_character_prompt_reasoning_w_fix_cognitive_wo_self(
                    #         book_title, character, character_profile, plot["summary"],
                    #         conversation["scenario"], motivation, thoughtless=args.wo_thought,
                    #         other_character_profiles=involved_character_profiles,
                    #         exclude_plot_summary=True, fixed_template=True,
                    #         add_output_example=add_output_example, add_rag=retrieval
                    #     )

                    elif args.cognitive_method == 'fixed_with_updates':
                        # TODO 待修改
                        system_prompt = get_character_prompt_reasoning_w_fix_cognitive_updates(
                            book_title, character, character_profile, plot["summary"],
                            conversation["scenario"], motivation, thoughtless=args.wo_thought,
                            other_character_profiles=involved_character_profiles,
                            exclude_plot_summary=True, fixed_template=True,
                            add_output_example=add_output_example, add_rag=retrieval,
                            previous_cognitive_result=character_cognitive_states[character]
                        )

                else:
                    system_prompt = get_character_prompt(
                        book_title, character, character_profile, plot["summary"],
                        conversation["scenario"], motivation, thoughtless=args.wo_thought,
                        other_character_profiles=involved_character_profiles,
                        exclude_plot_summary=True, fixed_template=True,
                        add_output_example=add_output_example, add_rag=retrieval
                    )
                    
                if args.cognitive_method == "sft_cognitive":
                    system_prompt = get_character_prompt_sft_cognitive(
                        book_title, character, character_profile, plot["summary"],
                        conversation["scenario"], motivation, thoughtless=args.wo_thought,
                        other_character_profiles=involved_character_profiles,
                        exclude_plot_summary=True,
                        fixed_template=False,
                        add_output_example=add_output_example, add_rag=retrieval
                    )

            if character not in special_characters:
                character_model = actor_model
            elif character == ENVIRONMENT:
                character_model = env_model
            elif character == NSP:
                character_model = nsp_model
            else:
                raise ValueError(f'Invalid character: {character}')

            if args.cognitive_method == 'two_stage':
                character_agent = Agent(
                    character_model, character, character_database,
                    system_prompt=system_prompt,
                    retrieval_target=retrieval if (retrieval and character not in special_characters) else None,
                    cognitive_system_prompt=cognitive_system_prompt,
                    system_prompt_params=system_prompt_params
                )
            else:
                character_agent = Agent(
                    character_model, character, character_database,
                    system_prompt=system_prompt,
                    retrieval_target=retrieval if (retrieval and character not in special_characters) else None
                )

            if args.cognitive_method == "sft_cognitive" and character not in special_characters:
                character_agent.merge_history = True
            else:
                character_agent.update('user', "===Conversation Start===\n\n")
            character_agents[character] = character_agent

        # Begin conversation simulation
        max_rounds = 20
        agent_conversations = []
        current_speaker = speaking_characters_w_env[0]  # Start with first character
        
        # Main conversation loop
        for i_round in range(max_rounds):
            if current_speaker == "<END CHAT>":
                break

            logger.info(f'===Round {i_round}===\n')
            
            # Generate responses for current speaker and get next speaker prediction
            for actor in [current_speaker, "NSP"]:
                current_agent = character_agents[actor]
                from utils import add_speaker_name
                
                if args.cognitive_method == 'fixed_with_updates' and actor != "NSP" and actor != ENVIRONMENT:
                    if i_round > 0 and character_cognitive_states[actor] is not None:  # 不是首轮且有认知状态
                        new_system_prompt = get_character_prompt_reasoning_w_fix_cognitive_updates(
                            book_title, actor, involved_character_profiles.get(actor, ''), plot["summary"],
                            conversation["scenario"],
                            find_motivation[0] if find_motivation else '',  # motivation
                            thoughtless=args.wo_thought,
                            other_character_profiles=involved_character_profiles,
                            exclude_plot_summary=True, fixed_template=True,
                            add_output_example=add_output_example, add_rag=retrieval,
                            previous_cognitive_result=character_cognitive_states[actor]
                        )
                        
                        current_agent.update_system_prompt(new_system_prompt)
                        logger.info(f"Updated system prompt for {actor} with previous cognitive state")


                if args.continue_from > i_round:
                    if actor == current_speaker:
                        response = conversation['dialogues'][i_round]['message']
                    else:  # NSP
                        response = conversation['dialogues'][i_round+1]['character'] if i_round < len(conversation['dialogues']) - 1 else '<END CHAT>'
                else:
                    response = current_agent.chat()
                    is_safe = test_safe(response)
                    try_count = 0
                    while not is_safe and try_count < 10:
                        response = current_agent.chat(temperature=0.8, top_k=5)
                        is_safe = test_safe(response)
                        try_count += 1

                if actor == "NSP":
                    # Process next speaker prediction
                    next_actor = response.split(':')[0].strip() if ':' in response else response

                    # Validate and set next speaker
                    if next_actor == "<END CHAT>" and i_round >= 5:
                        current_speaker = "<END CHAT>"
                    elif next_actor in speaking_characters_w_env and next_actor != current_speaker:
                        current_speaker = next_actor
                    else:
                        # Fallback to random selection if prediction is invalid
                        candidates = set(major_characters + [ENVIRONMENT]) - {current_speaker}
                        if not candidates:
                            candidates = set(speaking_characters_w_env) - {current_speaker}
                        candidates = list(candidates)
                        current_speaker = random.choice(candidates)
                    
                    logger.info(f"Next speaker: {current_speaker} (Raw response: {response})")
                    agent_conversations.append({"role": actor, "content": next_actor})
                    current_agent.update('assistant', next_actor)
                
                # else:
                #     # Process character/environment response
                #     response = add_speaker_name(response, actor)
                #     logger.info(f"{env_model if actor == ENVIRONMENT else actor_model}: {response}\n")
                #     agent_conversations.append({"role": actor, "content": response})

                #     # Update conversation history for all agents
                #     for other_actor, other_agent in character_agents.items():
                #         if other_actor == actor:
                #             other_agent.update('assistant', response)
                #         else:
                #             other_agent.update('user', remove_inner_thoughts(response))
                else:
                    original_response = response
                    thinking = ""
                    if actor != ENVIRONMENT:
                        if args.cognitive_method == "sft_cognitive":
                            # tag = "cognitive"
                            tag = "think"
                            is_cogdual=True
                        else:
                            tag = "think"
                            is_cogdual=False
                        response, thinking = parse_reasoning_response(response, tag=tag, is_cogdual=is_cogdual)

                    response_with_name = add_speaker_name(response, actor)
                    logger.info(f"{env_model if actor == ENVIRONMENT else actor_model}: {response_with_name}\n")

                    cognitive_response = current_agent.last_cognitive_response if args.use_reasoning else ""
                    agent_conversations.append({
                        "role": actor,
                        "content": response_with_name,
                        "original_response": original_response if args.use_reasoning else "",
                        "thinking": thinking if args.use_reasoning else "",
                        "cognitive_response": cognitive_response
                    })

                    if args.cognitive_method == 'fixed_with_updates':
                        character_cognitive_states[actor] = thinking

                    for other_actor, other_agent in character_agents.items():
                        if other_actor == actor:
                            other_agent.update('assistant', response_with_name)
                        else:
                            other_agent.update('user', remove_inner_thoughts(response_with_name))

        # Store simulation results
        results.append({
            'book_title': book_title,
            'i_p': i_p,
            'i_c': i_c,
            'circumstance': circumstance,
            'simulation': agent_conversations,
            'involved_character_profiles': involved_character_profiles
        })

    # Save simulation results
    os.makedirs(os.path.dirname(simulation_path), exist_ok=True)
    with open(simulation_path, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def gca_judging(test_file, actor_model, retrieval, judge_model, nth_exp=0):
    """
    Evaluates the quality of Given-Circumstance Acting (GCA) simulation results using multiple metrics.
    
    This function loads simulation results and evaluates them against reference dialogues using both automated metrics (BLEU, ROUGE-L) and LLM-based judgments across four dimensions:
    - Storyline Consistency: Measures alignment between the simulated conversation and original dialogue 
    - Anthropomorphism: Evaluates whether RPLAs behave in a human-like manner
    - Character Fidelity: Assesses whether RPLAs faithfully portray their characters
    - Storyline Quality: Evaluates whether the simulated conversation develops naturally

    Args:
        test_file (str): Path to JSON file containing test cases
        actor_model (str): Model name for character role-playing agents
        retrieval (str, optional): Type of retrieval data to enhance role-playing. Defaults to None (no retrieval).
        judge_model (str): Model name for LLM Judges.
        nth_exp (int, optional): Experiment ID.

    Returns:
        tuple: (avg_scores, cases)
            - avg_scores (dict): Average scores across all evaluation metrics
            - cases (dict): Detailed evaluation results for each test case
    """
    from utils import set_cache_path

    # Set up caching file for model outputs
    cache_path = f'.cache/{actor_model}.pkl'
    if nth_exp > 0:
        cache_path = f'{cache_path}-repeat={nth_exp}'
    set_cache_path(cache_path)
    
    # Configure paths based on model and retrieval settings
    actor_setting = f'{actor_model}{"_" + suffix}{"_rag=" + retrieval if retrieval else ""}'
    simulation_path = f'exp/simulation/{test_file.split("/")[-1].replace(".json", "")}_{actor_setting}.json'
    evaluation_path = simulation_path.replace('/simulation/', '/evaluation/')

    logger.info(f'Evaluating GCA Simulation for {actor_setting} on {test_file}\n\nThe results will be saved to {evaluation_path}')
    
    # Return cached evaluation results if available
    if os.path.exists(evaluation_path) and not (args.regenerate or args.reevaluate):
        res = json.load(open(evaluation_path, 'r'))
        return res['scores'], res['cases']
    
    # Load the simulation results
    simulation_results = json.load(open(simulation_path, 'r'))

    # Define evaluation dimensions
    dimensions = ['Storyline Consistency', 'Anthropomorphism', 'Character Fidelity', 'Storyline Quality']
    scores = { d: [] for d in dimensions + ['bleu', 'rouge_l'] }
    cases = {}

    # Evaluate each simulation result
    for result in simulation_results:
        book_title, i_p, i_c, circumstance, simulation = result['book_title'], result['i_p'], result['i_c'], result['circumstance'], result['simulation'] 
        
        # Verify indices match
        assert i_p == circumstance['plot']['i_p']
        assert i_c == circumstance['i_c']

        logger.info(f'Book {book_title}')

        # Filter out NSP messages and clean up simulation/reference for comparison
        simulation = result['simulation']
        simulation = [m for m in simulation if m['role'] != NSP]
        reference = circumstance['dialogues']

        # Remove inner thoughts for fair comparison
        simulation = [ m if m['role'] == ENVIRONMENT else 
            {**m, 'content': remove_inner_thoughts(m['content'])} 
            for m in simulation  ]

        reference = [ m if m['character'] == ENVIRONMENT else 
            {**m, 'message': remove_inner_thoughts(m['message'])} 
            for m in reference  ]

        # Convert to readable string format for evaluation
        simulation_str = '\n\n'.join([m['content'].strip('\n') for m in simulation])
        reference_str = '\n\n'.join([f"{m['character']}: {m['message']}".strip('\n') for m in reference])
            
        logger.info(f'===Simulation of {actor_setting}===\n\n**************\n{simulation_str}\n\n**************\n\n===Reference===\n\n**************\n{reference_str}\n\n**************\n\n')

        # Prepare context information for evaluation
        scenario_str =  circumstance['scenario']
        character_profile_str = '\n\n'.join([f"### {character}\n\n{profile.strip('')}" for character, profile in result['involved_character_profiles'].items()])
        major_characters = circumstance['major_characters']

        # Add special instructions for partial evaluation if needed
        additional_instructions = ''
        if args.continue_from > 0:
            additional_instructions = f'Please note that the first {args.continue_from} messages in the simulated conversation are the same as the reference. Focus your evaluation only on the content after these messages.'

        # Helper function to validate evaluation response format
        def parse_response(response, **kwargs):
            try:
                assert isinstance(response, dict)
                for k, v in response.items():
                    assert k in dimensions
                    assert 'flaws' in v

                    for f in v['flaws']:
                        if f.get('severity', None) is None:
                            f['severity'] = 1

                return response
            except:
                return False

        logger.info(f'{book_title}-{i_p}-{i_c}-{scenario_str}')

        # Count non-environment messages for score adjustment
        actor_rounds = len([m for m in simulation if m['role'] != ENVIRONMENT])
        eval_result = {}

        # Evaluate each dimension using LLM
        for dimension in dimensions:
            from prompts import critic_prompts
            critic_prompt = critic_prompts['self-play-deduct-template'].replace('{book}', book_title).replace('{plot_summary}', circumstance['plot']['summary']).replace('{scenario}', scenario_str).replace('{character_profiles}', character_profile_str).replace('{original_conversation}', reference_str).replace('{major_characters}', ', '.join(major_characters)).replace('{additional_instructions}', additional_instructions).replace('{dimension_name}', dimension).replace('{dimension_brief}', critic_prompts['dimension_details'][dimension]['dimension_brief']).replace('{dimension_criteria}', critic_prompts['dimension_details'][dimension]['dimension_criteria'])

            res = get_response_json([extract_json, parse_response], model=judge_model, messages=[{"role": "system", "content": critic_prompt}, {"role": "user", "content": simulation_str}])
            
            eval_result.update({dimension: res[dimension]})
            
            logger.info(json.dumps(res, ensure_ascii=False, indent=2)) 
            
            # Calculate dimension score with length penalty
            res[dimension]['score'] = max(0, min(100 - (sum([f['severity'] for f in res[dimension]['flaws'] if isinstance(f['severity'], int)]) - 0.3 * actor_rounds) * 5, 100) )

        # Calculate automated metrics
        bleu, rouge_l = calculate_bleu_rouge(reference[args.continue_from:], simulation[args.continue_from:])
        eval_result['bleu'] = bleu
        eval_result['rouge_l'] = rouge_l

        # Store evaluation results
        cases[f'{book_title}-{i_p}-{i_c}'] = {
            'simulation': simulation,
            'simulation_str': simulation_str,
            'score': sum([eval_result[dimension]['score'] for dimension in dimensions]) / len(dimensions),
            'critique': eval_result,
        }

        # Accumulate scores
        for dimension in dimensions:
            scores[dimension].append(eval_result[dimension]['score'])
        scores['bleu'].append(bleu)
        scores['rouge_l'].append(rouge_l)

    # Calculate average scores across all dimensions
    avg_scores = {dimension: sum(scores[dimension]) / max(1, len(scores[dimension])) for dimension in dimensions}
    avg_scores['avg'] = sum(avg_scores.values()) / len(avg_scores)
    avg_scores.update({metric: sum(scores[metric]) / max(1, len(scores[metric])) for metric in ['bleu', 'rouge_l']})

    logger.info(f'{actor_setting}: Average score of {len(simulation_results)} samples: \n{avg_scores["avg"]} {avg_scores} on {test_file}')

    # Save evaluation results
    os.makedirs(os.path.dirname(evaluation_path), exist_ok=True)
    with open(evaluation_path, 'w') as f:
        json.dump({'scores': avg_scores, 'cases': cases}, f, ensure_ascii=False, indent=2)

    return avg_scores, cases


if __name__ == "__main__":

    if args.nth_exp >= 0:
        nth_exps = [args.nth_exp]
    else:
        repeat_times = 3
        nth_exps = range(repeat_times)

    # Run experiments for each repeat
    for nth_exp in nth_exps:
        # Configure experiment name and logging
        exp_name = 'eval'
        if args.continue_from > 0: exp_name += f'-continue_from={args.continue_from}'    
        if nth_exp > 0: exp_name += f'-repeat={nth_exp}'
        
        logger = setup_logger(__name__, f'{__file__.split(".")[0]}-{exp_name}.log')

        # Initialize result storage
        all_cases = {} 
        all_scores = {} 

        from concurrent.futures import ProcessPoolExecutor
        import functools

        def generate(exp_args):
            """Run simulation for given experiment args"""
            actor_model, args, nth_exp = exp_args
        
            results = gca_simulation(
                args.test_file,
                actor_model, 
                args.env_model,
                args.nsp_model,
                args.retrieval,
                nth_exp
            )

            return results

        def evaluate(exp_args):
            """Run evaluation for given experiment args"""
            actor_model, args, nth_exp = exp_args

            scores, cases = gca_judging(
                args.test_file,
                actor_model,
                args.retrieval,
                args.judge_model,
                nth_exp
            )

            return scores, cases
        
        # List of actor models to evaluate
        actor_models = [args.actor_model] # you can modify the list to expand to multiple models

        # Create experiment args for each actor model
        exp_args = [(actor_model, args, nth_exp) for actor_model in actor_models]

        # Parallel execution path when multiple workers available
        if args.num_workers > 1 and len(exp_args) > 1:
            # First run all generate tasks simultaneously
            generate_futures = []
            with ProcessPoolExecutor(max_workers=args.num_workers) as generate_executor:
                for exp_arg in exp_args:
                    future = generate_executor.submit(generate, exp_arg)
                    generate_futures.append((future, exp_arg))
            
            # As generate tasks complete, run evaluate tasks in parallel
            with ProcessPoolExecutor(max_workers=args.num_workers) as evaluate_executor:
                evaluate_futures = []
                
                # Process completed generate tasks and submit evaluates
                for generate_future, exp_arg in generate_futures:
                    generate_future.result()  # Wait for generate to complete
                    future = evaluate_executor.submit(evaluate, exp_arg)
                    evaluate_futures.append((future, exp_arg))
                
                # Process evaluate results as they complete
                for evaluate_future, exp_arg in evaluate_futures:
                    scores, cases = evaluate_future.result()

                    actor_model = exp_arg[0]
                    # Create identifier for this model run
                    actor_setting = f'{actor_model}{"_" + suffix}{"_rag=" + args.retrieval if args.retrieval else ""}'

                    all_scores[actor_setting] = scores
                    all_cases[actor_setting] = cases

        # Sequential execution path
        else:
            for exp_arg in exp_args:
                generate(exp_arg)
                scores, cases = evaluate(exp_arg)

                actor_model = exp_arg[0]
                # Create identifier for this model run
                actor_setting = f'{actor_model}{"_" + suffix}{"_rag=" + args.retrieval if args.retrieval else ""}'

                all_scores[actor_setting] = scores
                all_cases[actor_setting] = cases
                
        # Log final results
        logger.info(f'Evaluation results:\n{json.dumps(all_scores, ensure_ascii=False, indent=2)}')