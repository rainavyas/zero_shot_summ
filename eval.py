import sys
import os
import argparse
import pandas as pd
from tqdm import tqdm
from rouge_score import rouge_scorer
from statistics import mean


if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--filepath', type=str, required=True, help='path to data with predicted summaries')
    commandLineParser.add_argument('--pred_name', type=str, default='ChatGPT Summary', help='column name for predicted summaries')
    commandLineParser.add_argument('--rouge', action='store_true', help='eval rouge score')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load data
    data = pd.read_csv(args.filepath)
    
    if args.rouge:
        rouge1 = []
        rouge2 = []
        rougeL = []
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        for _, row in tqdm(data.iterrows(), total=len(data)):
            score = scorer.score(row['Summary'], row[args.pred_name])
            rouge1.append(score['rouge1'][2])
            rouge2.append(score['rouge2'][2])
            rougeL.append(score['rougeL'][2])
    
    print(f'Rouge-1\t{mean(rouge1)}')
    print(f'Rouge-2\t{mean(rouge2)}')
    print(f'Rouge-L\t{mean(rougeL)}')

    
