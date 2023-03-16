import sys
import os
import argparse
import pandas as pd
from tqdm import tqdm
from rouge_score import rouge_scorer


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
            score = scorer.score(row['Summary'])
            import pdb; pdb.set_trace()

    
