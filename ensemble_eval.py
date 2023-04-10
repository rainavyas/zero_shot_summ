'''
Evaluate performance with multiple samples 
Select the 'average' sample if selection used as method
'''

import sys
import os
import argparse
import pandas as pd
from tqdm import tqdm
from rouge_score import rouge_scorer
from statistics import mean, stdev

def corpus_rouge(scorer, data, pred_col_name, ref_col_name='Summary'):
    rouge1 = []
    rouge2 = []
    rougeL = []
    
    for _, row in tqdm(data.iterrows(), total=len(data)):
        score = scorer.score(str(row[ref_col_name]), str(row[pred_col_name]))
        rouge1.append(score['rouge1'][2])
        rouge2.append(score['rouge2'][2])
        rougeL.append(score['rougeL'][2])
    return mean(rouge1), mean(rouge2), mean(rougeL)

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--filepath', type=str, required=True, help='path to data outputs with predicted summaries')
    commandLineParser.add_argument('--pred_name', type=str, default='gpt3.5_summary', help='base column name for predicted summaries')
    commandLineParser.add_argument('--rouge', action='store_true', help='eval rouge score')
    commandLineParser.add_argument('--selection', action='store_true', help='select the average sample')
    commandLineParser.add_argument('--num_seeds', type=int, default=10, help='number of summaries to consider')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/ensemble_eval.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load data
    data = pd.read_csv(args.filepath)

    if args.rouge:
        rouge1 = []
        rouge2 = []
        rougeL = []
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        for seed in range(1, args.num_seeds+1):
            print('seed', seed)
            pred_col_name = f'{args.pred_name}_seed{seed}'
            r1, r2, rL = corpus_rouge(scorer, data, pred_col_name)
            rouge1.append(r1)
            rouge2.append(r2)
            rougeL.append(rL)

        print('Single Model (averaged over multiple draws)')
        print(f'Rouge-1\t{mean(rouge1)}+-{stdev(rouge1)}')
        print(f'Rouge-2\t{mean(rouge2)}+-{stdev(rouge1)}')
        print(f'Rouge-L\t{mean(rougeL)}+-{stdev(rouge1)}')
        print()
    
    if args.selection:
        # Select the most average sample as per the rougeL metric
        selected_seed = []
        metric = 'rougeL'
        scorer = rouge_scorer.RougeScorer([metric], use_stemmer=True)
        for _, row in tqdm(data.iterrows(), total=len(data)):
            best = [None, 0] # [seed , rouge_score] 
            for i in range(1, args.num_seeds+1):
                total = 0
                for j in range(1, args.num_seeds+1):
                    score = scorer.score(str(row[f'{args.pred_name}_seed{j}']), str(row[f'{args.pred_name}_seed{i}']))
                    total += score[metric][2]
                if total > best:
                    best = [i, total]
            selected_seed.append(best[0])

        # Evaluate with the selected seed sample
        rouge1 = []
        rouge2 = []
        rougeL = []
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        for i, row in tqdm(data.iterrows(), total=len(data)):
            score = scorer.score(str(row['Summary']), str(row[f'{args.pred_name}_seed{selected_seed[i]}']))
            rouge1.append(score['rouge1'][2])
            rouge2.append(score['rouge2'][2])
            rougeL.append(score['rougeL'][2])

        print('Most Average Sample')
        print(f'Rouge-1\t{mean(rouge1)}')
        print(f'Rouge-2\t{mean(rouge2)}')
        print(f'Rouge-L\t{mean(rougeL)}')
        print()
                



    
