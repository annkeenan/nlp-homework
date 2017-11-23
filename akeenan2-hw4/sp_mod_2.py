import argparse
from collections import defaultdict, Counter
import math
import random
import os

alphabet = set()

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    
    if len(obs) < 2:
        return 'O'

    for st in states:
        for prev_st in states:
            new_obs = [obs[0], obs[1]]
            if obs[0] not in alphabet:
                new_obs[0] = '<unk>'
            if obs[1] not in alphabet:
                new_obs[1] = '<unk>'

            if (prev_st, st) in start_p:
                V[0][(prev_st, st)] = {'prob': start_p[(prev_st, st)] + emit_p[st][new_obs[1]] + emit_p[st][new_obs[0]], 'prev': None}
            else:
                V[0][(prev_st, st)] = {'prob': emit_p[st][new_obs[1]] + emit_p[st][new_obs[0]], 'prev': None}
    # Run Viterbi when t > 0
    for t in range(1, len(obs)-1):
        V.append({})
        for st in states:
            max_tr_prob = -math.inf
            for prev_st in states:
                for prev_prev_st in states:
                    if st in trans_p[(prev_prev_st, prev_st)] and (prev_prev_st, prev_st) in V[t-1] and V[t-1][(prev_prev_st, prev_st)]['prob'] + trans_p[(prev_prev_st, prev_st)][st] >= max_tr_prob:
                        max_tr_prob = V[t-1][(prev_prev_st, prev_st)]['prob'] + trans_p[(prev_prev_st, prev_st)][st]
                        if obs[t+1] in alphabet:
                            V[t][(prev_st, st)] = {'prob': max_tr_prob + emit_p[st][obs[t+1]], 'prev': (prev_prev_st, prev_st) }
                        else:
                            V[t][(prev_st, st)] = {'prob': max_tr_prob + emit_p[st]['<unk>'], 'prev': (prev_prev_st, prev_st)}
    # Return the most probable ordering
    opt = []
    max_prob = (-math.inf, None)
    for (prev_st, st), data in V[len(V)-1].items():
        if data['prob'] >= max_prob[0]:
            max_prob = (data['prob'], (prev_st, st))
    opt.extend([max_prob[1][0], max_prob[1][1]])
    prev_prev, prev = max_prob[1]
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][(prev_prev, prev)]['prev'][1])
        prev_prev, prev = V[t + 1][(prev_prev, prev)]['prev']
    return opt

def predict_data(test_file, states, start_p, trans_p, emit_p):
    accurate = 0
    total = 0

    with open(test_file+'.out', 'w') as wf:
        with open(test_file, 'r') as f:
            obs = []
            obs_tags = []
            for l, line in enumerate(f):
                try:
                    word, obs_tag = line.rstrip().split('\t', 1)
                except: # Newline
                    if not obs:
                        continue

                    # Predict tags for the sentence
                    pred_tags = viterbi(obs, states, start_p, trans_p, emit_p)
                    for i, word in enumerate(obs):
                        if obs_tags[i] == pred_tags[i]:
                            accurate += 1
                        wf.write('%s %s %s\n' % (obs[i], obs_tags[i], pred_tags[i]))
                    wf.write('\n')

                    # Reset arrays
                    obs = []
                    obs_tags = []
                    continue

                obs.append(word)
                obs_tags.append(obs_tag)
                total += 1
    return accurate/total

def train(train_file, train_data, states, start_p, trans_p, emit_p):
    accurate = 0
    total = 0
    with open(train_file+'.out', 'w') as wf:
        alphabet.clear()
        for line in random.sample(train_data, len(train_data)):
            obs_tags = []
            obs = []
            for l in line:
                obs.append(l[0])
                obs_tags.append(l[1])

            pred_tags = viterbi(obs, states, start_p, trans_p, emit_p)
            for i in range(len(obs)):
                wf.write('%s %s %s\n' % (obs[i], obs_tags[i], pred_tags[i]))

                if obs_tags[i] == pred_tags[i]:
                    accurate += 1
                total += 1

                # Update structured perceptron
                emit_p[obs_tags[i]][obs[i]] += 1
                emit_p[pred_tags[i]][obs[i]] -= 1

                if obs[i] not in alphabet: # Account for unknown words as well while training
                    emit_p[pred_tags[i]]['<unk>'] -= 1
                    emit_p[obs_tags[i]]['<unk>'] += 1
                    alphabet.add(obs[i])

                if i > 1:
                    trans_p[(pred_tags[i-2], pred_tags[i-1])][pred_tags[i]] -= 1
                    trans_p[(pred_tags[i-2], pred_tags[i-1])][pred_tags[i]] -= 1

            wf.write('\n')
    return accurate/total

def initialize(train_file, train_data, states, start_p, trans_p, emit_p):
    words = set()
    train_line = []
    prev_tag = None
    with open(train_file, 'r') as f:
        for l, line in enumerate(f):
            try:
                word, tag = line.rstrip().split('\t', 1)
                train_line.append((word, tag))
            except:
                train_data.append(train_line)
                train_line = []
                prev_tag = None
                continue
            states.add(tag)
            words.add(word)

            if prev_tag:
                start_p[(prev_tag, tag)] += 1
            else:
                prev_tag = tag

    for state in states:
        for prev_state in states:
            for prev_prev_state in states:
                trans_p[(prev_prev_state, prev_state)][state] = 0
        for word in words:
            emit_p[state][word] = 0
        emit_p[state]['<unk>'] = 0

if __name__ == '__main__':
    train_file = 'data/train'
    test_file = 'data/test'

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='train on file TRAIN')
    parser.add_argument('--test', help='test on file TEST')
    parser.add_argument('--output', help='output on file OUTPUT')
    parser.add_argument('-d', action='store_true', help='print debugging output')
    args = parser.parse_args()

    if args.train:
        train_file = args.train
    if args.test:
        test_file = args.test

    states = set()
    start_p = defaultdict(float)
    trans_p = defaultdict(dict) # dict of floats
    emit_p = defaultdict(dict)
    train_data = [] # store training data

    initialize(train_file, train_data, states, start_p, trans_p, emit_p)

    # Run through functions
    prev_accuracy = []
    training_accuracy = 0
    dev_accuracy = 0
    iterations = 0
    
    cmd = "perl conlleval.pl < %s.out | awk -F '; ' '{if (NR==2) {print $4}}'" % test_file
    while training_accuracy < 1:
        training_accuracy = train(train_file, train_data, states, start_p, trans_p, emit_p)
        dev_accuracy = predict_data(test_file, states, start_p, trans_p, emit_p)
        iterations += 1
        print('%d: train accuracy = %f, dev accuracy = %f' % (iterations, training_accuracy, dev_accuracy), end='')
        if args.d:
            os.system(cmd)
        print('')

        if prev_accuracy and iterations > 5 and training_accuracy < min(prev_accuracy):
            break

        prev_accuracy.append(training_accuracy)
        if len(prev_accuracy) > 5: # maintain five elements
            del prev_accuracy[0]
    cmd = "perl conlleval.pl < %s.out" % test_file
    os.system(cmd)
