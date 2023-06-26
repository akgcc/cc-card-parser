#####################
## IMPORTANT NOTE: ##
#####################
# This file is only used for testing purposes, the actual formula is calculated by the client and located in cc.js
import statistics
import numpy as np
import json
from copy import deepcopy
from pprint import pprint
import matplotlib.pyplot as plt

# current formula:
# U = uniqueness of an operator (% of clears they appear in)
# U' = all values of U (for every operator)
# S = standard deviation of U'
# M = max(U') / 2
# soul = weighted mean of U', where each weight = abs(M-U)/S

# Sub 18 clears are not used to calculate uniqueness.

# tweak 1:
# group all clears from the same doctor together and treat it as a single clear when generating U'
# this way a doctor won't lower their own soul rating by using the same team 3 times.
from itertools import chain, combinations
from collections import defaultdict
MAX_GROUP_SIZE = 5
def powerset(iterable,sizes=(1,MAX_GROUP_SIZE)):
    'sizes: tuple like (3,5), will include both bounds.'
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    # return [frozenset(i) for i in chain.from_iterable(combinations(s, r) for r in range(sizes[0],min(sizes[1]+1,len(s)+1)))]
    return [frozenset(i) for i in chain.from_iterable(combinations(s, r) for r in (1,3,4,5))]
def calculate_soul_test(data):
    'instead of considering only individual ops, also considers all possible combinations of ops (powerset)'
    # currently favours high-op clears WAY too much due to 12/13-op combos have the same "weight" as 1 or 2 op combos.
    tally = defaultdict(int)
    data_copy = defaultdict(set,deepcopy(data))
    for k,v in data.items():
        if 'duplicate_of' in v:
            # merge powersets togheter
            # data_copy[v['duplicate_of']]['squad'] = list( set(data_copy[v['duplicate_of']]['squad']) | set(v['squad']) )
            data_copy[v['duplicate_of']].setdefault('powerset', set())
            data_copy[v['duplicate_of']]['powerset'] |= set(powerset(v['squad']))
            del data_copy[k]
        else:
            data_copy[k].setdefault('powerset', set())
            data_copy[k]['powerset'] |= set(powerset(v['squad']))
    for k,v in data_copy.items():
        for subset in v['powerset']:
            tally[subset]+=1
    uniqueness = {k: (1 - (v-1)/len(data_copy)) for k,v in tally.items()} # ideally you use v-1 to ignore self but this will cause div by 0 problems
    # instead of SD based weights, just stretch the data out around the midpoint.
    # uniqueness = {k: 5 * ((1 - v/len(data_copy)) - .5) for k,v in tally.items()}
    print(len([v for k,v in uniqueness.items()]))
    sorted(uniqueness.items(),key=lambda x: x[1])[:len(uniqueness)]
    
    print('mean:',statistics.mean(uniqueness.values()))
    print("Standard Deviation of sample is",statistics.stdev(uniqueness.values()))
    # print(statistics.quantiles(uniqueness.values(),n=4,method='inclusive'))
    print('median:',statistics.median_grouped(uniqueness.values(),interval=1))
    print('median_h:',statistics.median_high(uniqueness.values()))
    MED_H = statistics.median_high(uniqueness.values())
    print('mean:',statistics.harmonic_mean(uniqueness.values()))
    print('max:',max(uniqueness.values()))
    print('percentile',np.percentile(list(uniqueness.values()), .1))
    first_percentile = np.percentile(list(uniqueness.values()), .1)
    pprint(len([(k,v) for k,v in uniqueness.items() if v<first_percentile]))
    # print([k for k,v in uniqueness.items() if v > first_percentile and len(k)==1])
    
    # filter to only sets with len 1 or very low uniqueness.
    uniqueness = {k:v if v<first_percentile or len(k)==1 else 0 for k,v in uniqueness.items()}
    # exit()
   
    
    # print(tally)
    # print(uniqueness)
    # print([v for v in uniqueness.values() if v<=0])
    # generate core ops:
    # core_sets = defaultdict(int)
    # for k,v in data.items():
        # for squad in powerset(v['squad'],(3,5)):
            # core_sets[squad] += 1
    # from pprint import pprint
    # pprint(sorted(core_sets.items(),key=lambda x: x[1],reverse=True)[:20])
    # most_soulless_set = max(core_sets.items(),key=lambda x: x[1])
    weights = {}
    #calculate weights per group size
    total_combos = len(powerset(range(1,14)))
    # for l in range(1,MAX_GROUP_SIZE+1):
    for l in (1,3,4,5):
        length_weight = total_combos / len(list(combinations(range(1,14), l)))
        
        d = {k:v for k,v in uniqueness.items() if len(k) == l}
        length_weight = (1 - len(d)/len(uniqueness))
        length_weight = 1 / len(list(combinations(range(1,14), l)))
        print(length_weight)
        vals = [v for v in d.values() if v]
        # length_weight = 1 
        SD = statistics.stdev(vals)
        MED = max(vals) / 2
        print('med,sd',MED,SD)
        
        print('mean:',statistics.mean(vals))
        print("Standard Deviation of sample is",statistics.stdev(vals))
        # print(statistics.quantiles(vals,n=4,method='inclusive'))
        print('median:',statistics.median_grouped(vals,interval=1))
        print('median_h:',statistics.median_high(vals))
        MED_H = statistics.median_high(vals)
        print('mean:',statistics.harmonic_mean(vals))
        print('max:',max(vals))
        # print('percentile',np.percentile([a for a in vals if a>MED_H], 75))
    
        # MED = statistics.median_high(vals)
        MED = statistics.harmonic_mean(vals)
        weights.update({k:v if v==0 else max(weights.values()) if l>1 else abs(v-MED)/SD * length_weight for k,v in d.items()})
        
        # weights[l] = len(list(combinations(range(1,14), l))) / total_combos
        # weights[l] = 14-l
        # weights[l] = len([k for k in uniqueness.keys() if len(k)==l]) / len(uniqueness.keys())
        # weights[l] = 2** (14-l)
        # possible combos of len l from 13 size squad. / ALL combinations from size 13
    # SD = statistics.stdev(uniqueness.values())
    # MED = max(uniqueness.values()) / 2
    # print('med,sd',MED,SD,max(uniqueness.values()),min(uniqueness.values()))
    # weights = {k:abs(v-MED)/SD for k,v in uniqueness.items()}
    # pprint(weights)
    
    
    
    # print([v for v in weights.values() if v<=0])
    for k,v in data.items():
        ps = powerset(v['squad'])
        t = [uniqueness[c]*weights[c] for c in ps]
        # t = [uniqueness[c]*weights[len(c)] for c in ps]
        sum_of_weights = sum([weights[k] for k in ps])
        # sum_of_weights = sum([weights[len(k)] for k in ps])
        # print([weights[k] for k in ps])
        # data[k]['soul'] = round(100 * (1 - sum(t) / sum_of_weights), 2)
        data[k]['soul'] = round(100 * (sum(t) / sum_of_weights), 2)
        # possible_combos = len(ps)
        # data[k]['soul'] = round(100 * (1 - sum(t) / possible_combos), 2)
        
        # avg without weights:
        # data[k]['soul'] = round(100 * sum([uniqueness[c] for c in ps]) / len(ps), 2)
    return data
def calculate_soul_old(data):
    # total = len(data) # this was incorrectly here.
    tally = {}
    data_stripped = deepcopy(data)
    for k,v in data_stripped.items():
        v['squad'] = [x['name'] for x in v['squad']]
    data_copy = deepcopy(data_stripped)
    for k,v in data_stripped.items():
        if 'duplicate_of' in v:
            data_copy[v['duplicate_of']]['squad'] = list( set(data_copy[v['duplicate_of']]['squad']) | set(v['squad']) )
            del data_copy[k]
    for k,v in data_copy.items():
        for charid in v['squad']:
            tally.setdefault(charid,0)
            if v['risk']>=18:
                tally[charid]+=1
    uniqueness = {k: 1 - v/len(data_copy) for k,v in tally.items()}
    # print(uniqueness)
    
    # generate core ops:
    # core_sets = defaultdict(int)
    # for k,v in data.items():
        # for squad in powerset(v['squad'],(3,5)):
            # core_sets[squad] += 1
    # from pprint import pprint
    # pprint(sorted(core_sets.items(),key=lambda x: x[1],reverse=True)[:20])
    # most_soulless_set = max(core_sets.items(),key=lambda x: x[1])
    
    print('mean:',statistics.mean(uniqueness.values()))
    print("Standard Deviation of sample is",statistics.stdev(uniqueness.values()))
    # print(statistics.quantiles(uniqueness.values(),n=4,method='inclusive'))
    print('median:',statistics.median_grouped(uniqueness.values(),interval=1))
    print('median_h:',statistics.median_high(uniqueness.values()))
    MED_H = statistics.median_high(uniqueness.values())
    print('mean:',statistics.harmonic_mean(uniqueness.values()))
    print('max:',max(uniqueness.values()))
    # print('percentile',np.percentile([a for a in uniqueness.values() if a>MED_H], 75))
    # print(uniqueness.values())
    SD = statistics.stdev(uniqueness.values()) or .0001
    MED = statistics.median_grouped(uniqueness.values())
    MED = max(uniqueness.values()) / 2
    weights = {k:abs(v-MED)/SD for k,v in uniqueness.items()}
    # print(weights)
    with open('character_table.json') as f:
        cmap = json.load(f)
    soulmap = {k:uniqueness[k]*weights[k] for k in weights.keys()}
    # for c in weights.keys():
        # print(f"{cmap[c]['name']}:\t{uniqueness[c]*weights[c]:.2f}")
    for k,v in sorted(soulmap.items(), key=lambda x: x[1]):
        print(f"{cmap[k]['name']+':':20}\t{v:.2f}\t{uniqueness[k]:.2f}\t{weights[k]:.2f}")
    for k,v in data.items():
        # t= [(uniqueness[c],)*int((1-uniqueness[c])/.2 + 1) for c in v['squad']]
        # t = list(sum(t, ())) # flatten
        squad = [x['name'] for x in v['squad']]
        t = [uniqueness[c]*weights[c] for c in squad]
        sum_of_weights = sum([weights[k] for k in squad])

        # squad_score = max([core_sets[s] for s in powerset(v['squad'],(3,5))],default=0)
        # print(sum(t))
        # t.append(squad_score/most_soulless_set[1]) # this experiment didn't go that well, it barely affects the data.
        # for squad in powerset(v['squad'],(3,5)):
            # squad_score += core_sets[squad]
        # print(squad_score)
        # data[k]['soul'] = round(100*(1-sum(t)/len(t)),2)
        try:
            data[k]['soul'] = round(100 * sum(t) / sum_of_weights, 2)
        except ZeroDivisionError:
            data[k]['soul'] = 100
    return data
    
from operator import mul
from functools import reduce 
def calculate_soul(data):
    import math
    with open('character_table.json') as f:
        cmap = json.load(f)
    tally = {}
    print({k:v for k,v in data.items() if 'risk' not in v})
    data_stripped = deepcopy({k:v for k,v in data.items() if v['risk']>=18})
    for k,v in data_stripped.items():
        v['squad'] = [x['name'] for x in v['squad']]
    data_copy = deepcopy(data_stripped)
    for k,v in data_stripped.items():
        if 'dupe_group' in v and v['risk']>=18:
            squad = data_copy.setdefault(v['dupe_group'],{'squad':[],'risk':18})['squad']
            data_copy[v['dupe_group']]['squad'] = list(set(squad) | set(v['squad']))
            del data_copy[k]
    for k,v in data_copy.items():
        for charid in v['squad']:
            tally.setdefault(charid,0)
            if v['risk']>=18:
                tally[charid]+=1
    # print(sorted(tally.values()))
    rms = statistics.mean([v**2 for v in tally.values()])**.5 # quadratic mean
    MIN_WEIGHT = .95
    ELITE_SOUL_EXEMPTIONS = ['char_214_kafka','char_497_ctable']#,'char_144_red','char_243_waaifu']
    ELITE_SOUL_SCALE = [-.5,.75,1]
    weights = {k:max(MIN_WEIGHT,abs(v-rms)/rms) for k,v in tally.items()}
    # weights = {k:max(MIN_WEIGHT,abs(v)/rms) for k,v in tally.items()}
    rarity_weights = [1,1,4,5,5,7.5]
    rarity_weights = [1,.5,4,4,5,6]
    rarity_weights = [1,1,1,1,1,1]
    weights = {k:v*rarity_weights[cmap[k]['rarity']] for k,v in weights.items()}
    uniqueness = {k: 1 - v/(len(data_copy)*1) for k,v in tally.items()}
    # print('tally',tally.values())
    uniqueness = {k: 1/(math.log(v+20)) for k,v in tally.items()}
    print(max(uniqueness.values()),min(uniqueness.values()))
    dif = 1 / max(uniqueness.values())
    uniqueness = {k: v*dif for k,v in uniqueness.items()}
    weights = {k:rarity_weights[cmap[k]['rarity']] for k,v in weights.items()}
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('uniqueness & weights')
    # ax1.plot(sorted(weights.values()))
    # ax1.plot(sorted(weights.values()))
    # ax2.plot(sorted([math.log(v,2) for v in weights.values()]))
    ax1.plot(sorted(uniqueness.values()))
    ax2.plot(sorted({k: 1 - v/(len(data_copy)*1) for k,v in tally.items()}.values()))
    # ax2.plot(sorted([1/(math.log(v,10)+1) for v in tally.values()]))
    # print(sorted(tally.values()))
    # print('uniq:',sorted(uniqueness.values()))
    
    math.e
    math.log
    # ax2.plot(sorted([math.log(v,10)+1 for v in tally.values()]))
    # print(sorted([math.log(v,2) for v in tally.values()]))
    # print(sorted(weights.values()))
    # ax2.plot([x[1] for x in sorted(weights.items(),key=lambda y: tally[y[0]])])
    # ax1.axhline(y=rms, color='r', linestyle='-')
    # plt.show()
    
    # soulmap = {k:uniqueness[k]*weights[k] for k in weights.keys()}
    # for k,v in sorted(soulmap.items(), key=lambda x: x[1]):
        # print(f"{cmap[k]['name']+':':30}\t{v:.2f}\t{uniqueness[k]:.2f}\t{weights[k]:.2f}")
    for c in cmap:
        if c not in weights:
            weights[c] = 0
            uniqueness[c] = 1
    for k,v in data.items():
        # squad = [x['name'] for x in v['squad']]
        # print(v['squad'])
        # len(cmap[k]['phases']) - data[
        # current - max? will scale from 0 to -2
        # do we want to have negative soul? maybe +1 so its 1 to -1 instead? only e0 of e2able ops gives negative??? this still lets u pad with 2* and 3*. (3* will have weight of 0 though, robots will default to negative soul)
                
        # print(k)
        def ss(c):
            return (1 if v['risk']==180 else ELITE_SOUL_SCALE[max(3+c['elite']-len(cmap[c['name']]['phases']),c['name'] in ELITE_SOUL_EXEMPTIONS)])
        t = [uniqueness[c['name']]**(weights[c['name']]*ss(c)) for c in v['squad']]
        sum_of_weights = sum([weights[c['name']]*ss(c) for c in v['squad']])
        try:
            # data[k]['soul'] = sum(t) / sum_of_weights #weighted
            # data[k]['soul'] = reduce(mul,[uniqueness[c] for c in squad],1) ** (1/len(squad)) # geometric
            # data[k]['soul'] = sum([1/uniqueness[c] for c in squad]) ** -1 # harmonic
            # data[k]['soul'] = sum([uniqueness[c] for c in squad]) /len(squad) # unweighted
            data[k]['soul'] = reduce(mul,[x for x in t],1) ** (1/sum_of_weights)
            # print(data[k]['soul'])
            data[k]['soul'] = round(100 * (data[k]['soul']), 2)
        except ZeroDivisionError:
            data[k]['soul'] = 100
    return data

if __name__ == '__main__':
    from time import time
    s = time()
    # for fn in ['json/data-ccbclear.json', 'json/data-cc0clear.json', 'json/data-cc1clear.json', 'json/data-cc2clear.json', 'json/data-cc3clear.json', 'json/data-cc4clear.json', 'json/data-cc5clear.json',
    # 'json/data-cc6clear.json', 'json/data-cc7clear.json']:
    for fn in ['json/data-cc7clear.json']:
        with open(fn,'r') as f:
            res = calculate_soul(json.load(f))
        with open(fn,'w') as f:
            json.dump(res,f)
    print('took',time()-s,'s')