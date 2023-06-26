# extract data directly from poll results source: var ANALYTICS_LOAD_DATA_
# examples (previous surveys):
#https://docs.google.com/forms/d/e/1FAIpQLSdOL5pjOH-VRV3Jg_AD20szZUWU-3e4h80aVpJ3HzM57qEeag/viewanalytics
#https://docs.google.com/forms/d/e/1FAIpQLSeJADbpZ6uPlkG9lNmoVaXD538EKiXEVspeFjjV6pa2q2c44A/viewanalytics
#https://docs.google.com/forms/d/e/1FAIpQLSe3ib2JNp5i7dQUArH2HfbFdVaVTeOPhgvMc1y6eqtZfWUMwQ/viewanalytics
import json
from pprint import pprint

# search for: #@ EDIT THIS in this file to find other changes you must make for a different survey.
RAW_RESULTS = 'poll_results_3_raw.json'
SKIP_QUESTIONS = [2054551172] # skip a specific question (the feedback q at the end)
BOOLEAN_QUESTIONS = ['Do you own this operator?','Promotion level','Skills and Modules']
SKIP_RESPONSES = ["Don't know"] # responses in this list will be ignored

OUTPUT_FILENAME = 'json/'+''.join(RAW_RESULTS.split('_raw'))
# will also output data to polldata.txt

with open(RAW_RESULTS,'r') as f:
    res = json.load(f)
QUESTIONS = res[0][1][1] # res[0][1][0] if no intro? (this is likely correct already)
ANSWERS = res[3]
QUESTION_MAP = {}
for q in QUESTIONS:
    question = q[1]
    if q[0] in SKIP_QUESTIONS:
        continue
    if len(q)>4 and q[4]:
        for e in q[4]:
            #example e: [135907008, [['1'], ['2'], ['3'], ['4'], ['5'], ['6'], ['7']], 1, ['Archetto'], None, None, None, None, None, None, None, [0]]
            try:
                QUESTION_MAP[e[0]] = (question,e[3][0])
            except IndexError:
                pass

ANSWER_MAP = {}
E2_ANSWER_MAP = {}
E2_COUNT_PER_OP = {}
bar_total = 0
# first get total survey count:
# print(QUESTION_MAP)
for a in ANSWERS:
    if a[0] in QUESTION_MAP:
        if QUESTION_MAP[a[0]][0] == 'Promotion level':
            total = sum([v[2] for v in a[1]])
            break
bar_total = total
for a in ANSWERS:
    if a[0] in QUESTION_MAP:
        metric,name = QUESTION_MAP[a[0]]
        metric = metric.strip()
        name = name.strip()
        # print(a)
        if metric not in BOOLEAN_QUESTIONS:
            # print(QUESTION_MAP[a[0]],a)
            m = ANSWER_MAP.setdefault(name, {})
            flat_results = [int(item) for sublist in [[v[0]]*v[2] for v in a[1] if v[0] not in SKIP_RESPONSES] for item in sublist]
            m[metric] = sum(flat_results)/len(flat_results)
        else:
            # total = sum([v[2] for v in a[1]]) # calc this ahead of time instead
            # bar_total = total
            _map = {v[0]:v[2] for v in a[1]}
            # print(_map)
            m = E2_ANSWER_MAP.setdefault(name, {})
            # m['Ownership'] = _map['Yes'] + _map['Yes and E2']
            # m['E2'] = _map['Yes and E2'] / total
            # m['Ownership'] /= total
            if metric == 'Promotion level':
                m['Ownership'] = _map['Own'] + _map['E2'] + _map['E2 90'] + _map.get('E1',0)
                m['E2'] = (_map['E2'] + _map['E2 90'] ) / total
                E2_COUNT_PER_OP[name] = (_map['E2'] + _map['E2 90'] )
                m['Ownership'] /= total
            if metric == 'Skills and Modules':
                m['S1'] = _map.get('S1M3',0) / total
                m['S2'] = _map.get('S2M3',0) / total
                m['S3'] = _map.get('S3M3',0) / total
            # for k,v in _map.items():
                # m[k] = v/total
#2nd pass as we need e2 total for this:
for a in ANSWERS:
    if a[0] in QUESTION_MAP:
        metric,name = QUESTION_MAP[a[0]]
        metric = metric.strip()
        name = name.strip()
        if metric == 'Skills and Modules':
            _map = {v[0]:v[2] for v in a[1]}
            m = E2_ANSWER_MAP.setdefault(name, {})
            m['S1'] = _map.get('S1M3',0) / E2_COUNT_PER_OP[name]
            m['S2'] = _map.get('S2M3',0) / E2_COUNT_PER_OP[name]
            m['S3'] = _map.get('S3M3',0) / E2_COUNT_PER_OP[name]
###################
# fix name errors:#
###################
# key = fake name (from survey)
# value = real name

#@ EDIT THIS
NAME_MAP = {
    '29': 'Surtr',
    'Kal\'tits': 'Kal\'tsit',
    'Ch\'en The Holungday': 'Ch\'en the Holungday',
    'Skadi The Corrupting Heart': 'Skadi the Corrupting Heart',
    'Pozemka':'Pozëmka',
}
for k,v in NAME_MAP.items():
    ANSWER_MAP.setdefault(v,{}).update(ANSWER_MAP.get(k,{}))
    ANSWER_MAP.pop(k,None)
    E2_ANSWER_MAP.setdefault(v,{}).update(E2_ANSWER_MAP.get(k,{}))
    E2_ANSWER_MAP.pop(k,None)

###################
###################
###################
# map questions to the axes of the chart
#@ EDIT THIS
col_map = {
'How good is the operator on CC high risk?':'CC',
'How good of a pick the operator is on IS2?':'IS2',
'How good is the operator on SSS?':'SSS',
}

for v in ANSWER_MAP.values():
    for k2, v2 in col_map.items():
        v[v2] = v[k2]
        del v[k2]
for v in ANSWER_MAP.values():
    v['Overall'] = sum(v.values())/len(v.values())

#@ EDIT THIS
# order = ['Power','Utility','Fun','Coom','Overall']
order = ['CC','IS2','SSS','Overall']

with open('polldata.txt','w') as f:
    f.write('Name\t'+'\t'.join(order)+'\n')
    for k,v in ANSWER_MAP.items():
        line = f'{k}\t'
        line+= "\t".join([str(v[n]) for n in order])
        f.write(line+'\n')
    f.write('\n\n')
    f.write('Name\t'+'\t'.join(['Ownership','E2','E2%OfOwners'])+'\n')
    for k,v in E2_ANSWER_MAP.items():
        line = f'{k}\t'
        line+= "\t".join([str(v[n]) for n in ['Ownership','E2']])
        line+= '\t'+str(v['E2']/v['Ownership'])
        f.write(line+'\n')
with open(OUTPUT_FILENAME,'w') as f:
    json.dump({"scatter":{"data":ANSWER_MAP,"default_axes":order}, "bar":{"data":E2_ANSWER_MAP,"total":bar_total},"date":"February 2023","url":"https://docs.google.com/forms/d/e/1FAIpQLSe3ib2JNp5i7dQUArH2HfbFdVaVTeOPhgvMc1y6eqtZfWUMwQ/viewanalytics"}, f)