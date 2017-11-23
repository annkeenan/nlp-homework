import argparse
from collections import defaultdict
from tree import Tree

class Grammar(object):
    def __init__(self, freq=False, prob=False):
        self.counts = defaultdict(int)
        self.rules = defaultdict(float)
        self.nonterminals = []
        self.terminals = set()
        self.freq = freq
        self.prob = prob

    def print_prob(self, rule, prob):
        if len(rule[1]) == 1:
            print('%s -> %s # %.3f' % (rule[0], rule[1][0], prob))
        else:
            print('%s -> %s %s # %.3f' % (rule[0], rule[1][0], rule[1][1], prob))

    def construct_cfg(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                tree = Tree.from_str(line.rstrip())
                self.process_cfg(tree.root)
        if self.freq:
            print('\n1a.')
            print('Number of unique rules = %d' % len(self.rules))
            print('Five most frequent rules:')
            for rule in sorted(self.rules, key=self.rules.get, reverse=True)[:5]:
                if len(rule[1]) == 1:
                    print('%s -> %s # %d' % (rule[0], rule[1][0], self.rules[rule]))
                else:
                    print('%s -> %s %s # %d' % (rule[0], rule[1][0], rule[1][1], self.rules[rule]))

        self.cond_prob()
        if self.prob:
            print('\n1b.')
            print('Grammar:')
            for rule, prob in self.rules.items():
                self.print_prob(rule, prob)
            print('Five highest-probability rules:')
            for rule in sorted(self.rules, key=self.rules.get, reverse=True)[:5]:
                self.print_prob(rule, self.rules[rule])
            print('')

        return self.rules

    def process_cfg(self, node):
        if len(node.children) != 0:
            self.counts[node.label] += 1
            children = [child.label for child in node.children]
            self.rules[(node.label, tuple(children))] += 1
            for child in node.children:
                self.process_cfg(child)
        else:
            self.terminals.add(node.label)

    def cond_prob(self):
        for rule, count in self.rules.items():
            self.rules[rule] = count / self.counts[rule[0]]
        self.nonterminals = list(self.counts.keys())

if __name__ == '__main__':
    trees_file = 'trees/train.trees.pre.unk'

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', action='store_true', help='print out the five most frequent rules')
    parser.add_argument('-p', action='store_true', help='print out the five highest-probability rules')
    parser.add_argument('--trees', help='use the trees in the specified file')
    args = parser.parse_args()

    if args.trees:
        trees_file = args.trees

    cfg = Grammar(args.f, args.p)
    cfg.construct_cfg(trees_file)
