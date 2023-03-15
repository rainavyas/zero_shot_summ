'''
ChatGPT used to summarise
'''

import openai
import sys
import os
import argparse
import pandas as pd
from tqdm import tqdm
import time

def content_merge(row):
    text = f'''Assessment: {row["Assessment"]}\n
           Subjective Section: {row["Subjective Sections"]}\n
           Objective Section: {row["Objective Sections"]}\n'''
    return text
        

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--data_path', type=str, default='../../bionlp2023-1a-train-og.csv', help='path to data')
    commandLineParser.add_argument('--out_dir', type=str, default='experiments/generated_summaries', help='path to dir to save output summaries')
    commandLineParser.add_argument('--sys_prompt_path', type=str, default='system_prompts.txt', help='select system prompt')
    commandLineParser.add_argument('--sys_prompt_ind', type=int, default=0, help='file for system prompts')
    commandLineParser.add_argument('--user_prompt_path', type=str, default='user_prompts.txt', help='file for user prompts')
    commandLineParser.add_argument('--user_prompt_ind', type=int, default=0, help='select user prompt')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/summarise_chatgpt.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Read the whole dataset
    data = pd.read_csv(args.data_path)

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
        time.sleep(3.1) # necessary for open ai rate limit
    
    data['ChatGPT Summary'] = gpt_summs

    # save data
    out_file = f'{args.out_dir}/chatgpt_system{args.sys_prompt_ind}_user{args.user_prompt_ind}.csv'
    data.to_csv(out_file)
    

