'''
Given multiple submission files, select most average sample and save output file
'''

import sys
import os
import argparse
from tqdm import tqdm
from rouge_score import rouge_scorer

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('--filepaths', type=str, nargs='+', required=True, help='path to data outputs with predicted summaries')
    commandLineParser.add_argument('--outfile', type=str, required=True, help='path to save final predictions')
    args = commandLineParser.parse_args()

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/sample_select_test.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')


    # load data
    data = []
    for fpath in args.filepaths:
        with open(fpath, 'r') as f:
            summ = f.readlines()
        summ = [s.strip('\n') for s in summ]
        data.append(summ)
    
    # select samples
    selected_sample = []
    metric = 'rougeL'
    scorer = rouge_scorer.RougeScorer([metric], use_stemmer=True)
    for samples in tqdm(zip(*data)):
        best = [None, 0] # [seed , rouge_score] 
        for i in range(len(samples)):
            total = 0
            for j in range(len(samples)):
                score = scorer.score(samples[j], samples[i])
                total += score[metric][2]
            if total > best[1]:
                best = [i, total]
        selected_sample.append(best[0])
    
    # save selected samples
    with open(args.outfile, 'w') as f:
        for sample in selected_sample:
            f.write(sample+'\n')


    

