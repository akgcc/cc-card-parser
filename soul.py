import statistics
import numpy as np
import json
from copy import deepcopy
from pprint import pprint
# current formula:
# U = uniqueness of an operator (% of clears they appear in)
# U' = all values of U (for every operator)
# S = standard deviation of U'
# M = max(U') / 2
# soul = weighted mean of U', where each weight = abs(M-U)/S

# tweak 1:
# group all clears from the same doctor together and treat it as a single clear when generating U'
# this way a doctor won't lower their own soul rating by using the same team 3 times.

def calculate_soul(data):
    total = len(data)
    tally = {}
    data_copy = deepcopy(data)
    for k,v in data.items():
        if 'duplicate_of' in v:
            data_copy[v['duplicate_of']]['squad'] = list( set(data_copy[v['duplicate_of']]['squad']) | set(v['squad']) )
            del data_copy[k]
    for k,v in data_copy.items():
        for charid in v['squad']:
            tally.setdefault(charid,0)
            tally[charid]+=1    
    uniqueness = {k:v/total for k,v in tally.items()}
    # print(uniqueness)

    print('mean:',statistics.mean(uniqueness.values()))
    print("Standard Deviation of sample is",statistics.stdev(uniqueness.values()))
    # print(statistics.quantiles(uniqueness.values(),n=4,method='inclusive'))
    print('median:',statistics.median_grouped(uniqueness.values(),interval=1))
    print('median_h:',statistics.median_high(uniqueness.values()))
    MED_H = statistics.median_high(uniqueness.values())
    print('mean:',statistics.harmonic_mean(uniqueness.values()))
    print('max:',max(uniqueness.values()))
    print('percentile',np.percentile([a for a in uniqueness.values() if a>MED_H], 75))
    # print(uniqueness.values())
    SD = statistics.stdev(uniqueness.values())
    MED = statistics.median_grouped(uniqueness.values())
    MED = max(uniqueness.values()) / 2
    weights = {k:abs(v-MED)/SD for k,v in uniqueness.items()}
    # print(weights)
    for k,v in data.items():
        # t= [(uniqueness[c],)*int((1-uniqueness[c])/.2 + 1) for c in v['squad']]
        # t = list(sum(t, ())) # flatten
        
        t = [uniqueness[c]*weights[c] for c in v['squad']]
        sum_of_weights = sum([weights[k] for k in v['squad']])

        # data[k]['soul'] = round(100*(1-sum(t)/len(t)),2)
        
        data[k]['soul'] = round(100 * (1 - sum(t) / sum_of_weights), 2)
    return data
if __name__ == '__main__':
    for fn in ['data-ccbclear.json', 'data-cc0clear.json', 'data-cc1clear.json', 'data-cc2clear.json', 'data-cc3clear.json', 'data-cc4clear.json']:
        with open(fn,'r') as f:
            res = calculate_soul(json.load(f))
        with open(fn,'w') as f:
            json.dump(res,f)