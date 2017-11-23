import argparse
from collections import defaultdict
import math

alphabet = set()

def dptable(V, states):
    # Print a table of steps from dictionary
    yield "%13s  " % "" + " ".join(("%10s" % i) for i in range(len(V)))
    for state in states:
        yield "%13s: " % state + " ".join("%-10s" % ("%f" % v[state]["prob"]) for v in V if state in v)

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    for st in states:
        if obs[0] in emit_p[st]:
            V[0][st] = {'prob': math.log(start_p[st], 10) + math.log(emit_p[st][obs[0]], 10), 'prev': None}
        else:
            V[0][st] = {'prob': math.log(start_p[st], 10) + math.log(emit_p[st]['<unk>'], 10), 'prev': None}

    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})
        for st in states:
            max_tr_prob = -math.inf
            for prev_st in states:
                if st in trans_p[prev_st] and V[t-1][prev_st]['prob'] + math.log(trans_p[prev_st][st], 10) >= max_tr_prob:
                    max_tr_prob = V[t-1][prev_st]['prob'] + math.log(trans_p[prev_st][st], 10)
                    if obs[t] in emit_p[st]:
                        V[t][st] = {'prob': max_tr_prob + math.log(emit_p[st][obs[t]], 10), 'prev': prev_st}
                    else:
                        V[t][st] = {'prob': max_tr_prob + math.log(emit_p[st]['<unk>'], 10) + math.log(start_p[st]), 'prev': prev_st}

    # Debug
    # for line in dptable(V, states):
    #     print(line)

    # Return the most probable ordering
    opt = []
    prev = None
    
    max_prob = (-math.inf, None)
    for st, data in V[len(V)-1].items():
        if data['prob'] >= max_prob[0]:
            max_prob = (data['prob'], st)
    
    opt.append(max_prob[1])
    prev = max_prob[1]
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][prev]['prev'])
        prev = V[t + 1][prev]['prev']
    return opt

def predict_data(test_file, states, start_p, trans_p, emit_p):
    with open(test_file+'.out', 'w') as wf:
        with open(test_file, 'r') as f:
            obs = []
            obs_tags = []
            n = 0 # track the sentence number
            for line in f:
                try:
                    word, obs_tag = line.rstrip().split('\t', 1)
                except: # Newline
                    # Predict tags for the sentence
                    if not obs:
                        continue
                    pred_tags = viterbi(obs, states, start_p, trans_p, emit_p)
                    for i in range(len(obs)):
                        if n < 5:
                            print('%s %s %s' % (obs[i], obs_tags[i], pred_tags[i]))
                        wf.write('%s %s %s\n' % (obs[i], obs_tags[i], pred_tags[i]))
                    if n < 5:
                        print('')
                    wf.write('\n')
                    
                    # Reset arrays
                    obs = []
                    obs_tags = []
                    n += 1
                    continue
                    
                obs.append(word)
                obs_tags.append(obs_tag)

def print_probabilities(trans_p, emit_p):
    print('tag bigram probabilities')
    print('p(B-person | O) = %f' % (trans_p['O']['B-person']))
    print('p(O | O) = %f' % (trans_p['O']['O']))
    for dest_tag in ['B-person', 'I-person']:
        if dest_tag in trans_p['B-person']:
            print('p(%s | B-person) = %f' % (dest_tag, trans_p['B-person'][dest_tag]))
    for dest_tag in ['B-person', 'I-person','O']:
        if dest_tag in trans_p['I-person']:
            print('p(%s | I-person) = %f' % (dest_tag, trans_p['I-person'][dest_tag]))
    print('-'*80)

    print('tag-word probabilities')
    for word in ['God', 'Justin', 'Lindsay']:
        for tag in ['B-person', 'O']:
            try:
                print('p(%s | %s) = %f' % (word, tag, emit_p[tag][word]))
            except:
                print('p(%s | %s) = %f' % (word, tag, emit_p[tag]['<unk>']))
    print('-'*80)

def convert_probabilities(states, start_p, trans_p, emit_p):
    # Convert start_p
    total = sum(start_p.values())
    for tag in start_p.keys():
        start_p[tag] /= total

    # Convert trans_p
    for source_state, dest_states in trans_p.items():
        total = sum(trans_p[source_state].values())
        for dest_state in dest_states.keys():
            trans_p[source_state][dest_state] /= total

    # Smooth and convert emit_p
    add = 0.1
    for state in states:
        emit_p[state]['<unk>'] = 0
        total = sum(emit_p[state].values()) + add*len(emit_p[state])
        for word, wt in emit_p[state].items():
            emit_p[state][word] = (wt+add)/total

def read_data(train_file, states, start_p, trans_p, emit_p):
    num_tokens = 0
    prev_tag = None

    with open(train_file) as f:
        for line in f:
            try:
                word, tag = line.rstrip().split('\t',1)
            except: # Newline
                prev_tag = None
                continue

            alphabet.add(word)
            states.add(tag)

            # Track tag|word counts
            if word not in emit_p[tag]:
                emit_p[tag][word] = 0.
            emit_p[tag][word] += 1

            # Track tag|tag counts
            if not prev_tag:
                prev_tag = tag
                start_p[tag] += 1
            else:
                if tag not in trans_p[prev_tag]:
                    trans_p[prev_tag][tag] = 0.
                trans_p[prev_tag][tag] += 1
                start_p[tag] += 1

            # Increment counter
            num_tokens += 1
            prev_tag = tag

    # Output the counter values
    print('training data types')
    print('# tokens\t%d' % num_tokens)
    print('# tag types\t%d' % len(states))
    print('# word types\t%d' % len(alphabet))
    print('-'*80)

if __name__ == '__main__':
    train_file = 'data/train'
    test_file = 'data/test'

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='train on file TRAIN')
    parser.add_argument('--test', help='test on file TEST')
    parser.add_argument('--output', help='output on file OUTPUT')
    args = parser.parse_args()

    if args.train:
        train_file = args.train
    if args.test:
        test_file = args.test

    states = set()
    start_p = defaultdict(float)
    trans_p = defaultdict(dict) # dict of floats
    emit_p = defaultdict(dict)

    # Run through functions
    read_data(train_file, states, start_p, trans_p, emit_p)
    convert_probabilities(states, start_p, trans_p, emit_p)
    print_probabilities(trans_p, emit_p)
    predict_data(test_file, states, start_p, trans_p, emit_p)
