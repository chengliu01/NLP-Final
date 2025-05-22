import json
import json
from pathlib import Path
from typing import Union, Dict, List
import random
random.seed(42)
from datasets import load_dataset

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    print(f"Loaded JSON data from {file_path}:")
    print(len(data), " items")
    return data


def load_hf_dataset(local_dir=None):
    from huggingface_hub import hf_hub_download
    file_path = hf_hub_download(
        repo_id="Neph0s/CoSER",
        filename="train/sft_conversations_sharegpt.json",
        repo_type="dataset",
        local_dir=local_dir
    )
    print(f"文件已下载到: {file_path}")
    return file_path


def download_and_save(download_dir, save_path):
    import shutil
    import os
    file_path = load_hf_dataset(local_dir=download_dir)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    shutil.copy(file_path, save_path)
    print(f"文件已保存到: {save_path}")
    return save_path


def random_samples(fp_in: str, fp_out:str=None, limit: int = 10, save: bool = True):
    with open(fp_in, "r") as f:
        data = json.load(f)
    print(f"Train size:{len(data)}")
    sub_data = random.sample(data, limit)
    if save:
        with open(fp_out, "w") as f_out:
            json.dump(sub_data, f_out, ensure_ascii=False, indent=4)
    return sub_data

def show_conversation(messages:list):
    system_content = messages[0]["value"]
    print("===system content==\n{}".format(system_content))
    conversation_str = ""
    for message in messages[1:]:
        conversation_str += f"from {message['from']}:\n{message['value']}"
    print(f"===conversation===\n{conversation_str}")


if __name__ == "__main__":

    # download_path = "/apdcephfs_cq10/share_1567347/share_info/sorenliu/code/rl_code/verl/data/coser_trainset"
    # file_path = load_hf_dataset(local_dir=download_path)

    coser_train_fp = "/apdcephfs_cq10/share_1567347/share_info/sorenliu/code/rl_code/verl/data/coser_trainset/train/sft_conversations_sharegpt.json"

    import time
    s = time.time()
    sub_data = random_samples(
        coser_train_fp,
        fp_out=None,
        limit=1000,
        save=False
    )
    e = time.time()
    print(f"Execution time: {e - s} seconds")


   