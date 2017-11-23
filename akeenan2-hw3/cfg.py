import argparse
from math import log
from tree import Node, Tree
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt

from grammar import Grammar

START = 'TOP'

class CKY(object):
    def __init__(self, cfg, verbose=False, plot=False):
        self.cfg = cfg
        self.start = cfg.nonterminals.index(START)
        self.verbose = verbose
        self.plot = plot

    def parse_sentences(self, input_filename, output_filename):
        with open(input_filename, 'r') as input_file:
            if self.verbose:
                for string in list(input_file)[:5]:
                    string = string.rstrip().split(' ')
                    tree = self.viterbi_parser(string)
                    if tree:
                        print('%s\t%.5f' % (str(tree[0]), tree[1]))
                    else:
                        print('\n')
            else:
                sentence_lengths = []
                parsing_times = []
                with open(output_filename, 'w') as output_file:
                    for string in input_file:
                        string = string.rstrip().split(' ')
                        sentence_lengths.append(log(len(string)))

                        start_time = datetime.now()
                        tree = self.viterbi_parser(string)
                        end_time = datetime.now()
                        elapsed_time = end_time - start_time

                        parsing_times.append(log(elapsed_time.microseconds))
                        if tree:
                            output_file.write('%s\n' % str(tree[0]))
                        else:
                            output_file.write('\n')
                if self.plot:
                    plt.plot(sentence_lengths, parsing_times, 'bo')
                    plt.ylabel('parsing time (log(microseconds))')
                    plt.xlabel('sentence length (log(words))')
                    plt.show()

    def viterbi_parser(self, string):
        n = len(string)
        r = len(self.cfg.nonterminals)
        chart = [[defaultdict(int) for k in range(r)] for j in range(n+1)]

        for i in range(1,n+1):
            if string[i-1] not in self.cfg.terminals:
                string[i-1] = '<unk>'
            for rule, prob in self.cfg.rules.items():
                prob = log(prob, 10) # convert to log probability
                if rule[1][0] == string[i-1]:
                    chart[i][1][rule[0]] = (
                        prob, (
                            (string[i-1], None),
                        )
                    )

        start_state = None
        for l in range(2, n+1):
            for s in range(1, n-l+2):
                for p in range(1, l):
                    for rule, prob in self.cfg.rules.items():
                        if len(rule[1]) == 2 and \
                            rule[1][0] in chart[s][p] and \
                            rule[1][1] in chart[s+p][l-p]:
                            prob = \
                                chart[s][p][rule[1][0]][0] + \
                                chart[s+p][l-p][rule[1][1]][0] + \
                                log(prob, 10) # convert to log probability

                            if rule[0] not in chart[s][l] or \
                                prob > chart[s][l][rule[0]][0]:
                                chart[s][l][rule[0]] = (
                                    prob, (
                                        (rule[1][0], (s,p)),
                                        (rule[1][1], (s+p,l-p))
                                    )
                                )
        if START in chart[1][n]:
            return self.construct_tree(chart, chart[1][n][START])
        else:
            return None

    def construct_tree_recurse(self, chart, node, state):
        for pointer in state[1]:
            child = Node(pointer[0],[])
            node.append_child(child)
            if pointer[1]:
                self.construct_tree_recurse(chart, child, chart[pointer[1][0]][pointer[1][1]][pointer[0]])

    def construct_tree(self, chart, state):
        start = Node(START, [])
        self.construct_tree_recurse(chart, start, state)
        return (Tree(start), state[0])

if __name__ == '__main__':
    trees_file = 'trees/train.trees.pre.unk'
    strings_file = 'strings/dev.strings'
    output_file = 'output/dev.parses'

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', action='store_true', help='print the first five parses from parsing strings')
    parser.add_argument('-g', action='store_true', help='show a plot of the parsing time vs sentence length')
    parser.add_argument('--trees', help='use the trees in the specified file')
    parser.add_argument('--strings', help='use the strings in the specified file')
    parser.add_argument('--output', help='specify an output file')
    args = parser.parse_args()

    if args.trees:
        trees_file = args.trees
    if args.strings:
        strings_file = args.strings
    if args.output:
        output_file = args.output

    cfg = Grammar()
    cfg.construct_cfg(trees_file)
    cky = CKY(cfg, args.p, args.g)
    cky.parse_sentences(strings_file, output_file)
