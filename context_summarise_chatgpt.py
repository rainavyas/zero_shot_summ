'''
In context k-shot shot learning for ChatGPT
'''

import openai
import sys
import os
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np

from src.tools.tools import content_merge

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--data_path', type=str, default='../../bionlp2023-1a-train-og.csv', help='path to data')
    commandLineParser.add_argument('--folds_path', type=str, default='/home/alta/relevance/vr311/bionlp/split.npy', help='path to fold splits')
    commandLineParser.add_argument('--fold', type=int, default=0, help='select fold of data to evaluate upon')
    commandLineParser.add_argument('--out_dir', type=str, default='experiments/generated_summaries', help='path to dir to save output summaries')
    commandLineParser.add_argument('--sys_prompt_path', type=str, default='src/prompts/system_prompts.txt', help='select system prompt')
    commandLineParser.add_argument('--sys_prompt_ind', type=int, default=0, help='file for system prompts')
    commandLineParser.add_argument('--user_prompt_path', type=str, default='src/prompts/user_prompts_context.txt', help='file for user prompts')
    commandLineParser.add_argument('--user_prompt_ind', type=int, default=0, help='select user prompt')
    commandLineParser.add_argument('--seed', type=int, default=1, help='select seed for reproducibility')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/context_summarise_chatgpt.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Read the whole dataset
    data = pd.read_csv(args.data_path)

    # split into evaluation and training folds
    folds = np.load(args.folds_path)
    train = data.iloc[~folds[args.fold]]
    test = data.iloc[folds[args.fold]]
    import pdb; pdb.set_trace()

    # load prompts
    with open(args.sys_prompt_path, 'r') as f:
        sys_prompts = f.readlines()
    sys_prompt = sys_prompts[args.sys_prompt_ind].rstrip('\n') 

    with open(args.user_prompt_path, 'r') as f:
        user_prompts = f.readlines()
    user_prompt = user_prompts[args.user_prompt_ind].rstrip('\n') 

    gpt_summs = []
    for _, row in tqdm(data.iterrows(), total=len(data)):
        content = content_merge(row)
        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f'{user_prompt}\n{content}'}
        ]
        response = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=msgs)
        gpt_summs.append(response['choices'][0]['message']['content'])
        # time.sleep(3.1) # necessary for open ai rate limit
    
    data['ChatGPT Summary'] = gpt_summs