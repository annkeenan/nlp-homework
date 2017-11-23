from cfg import Grammar, CKY
import argparse
from tree import Tree
from collections import defaultdict

DEBUG = False
MOD = 1
FALLBACK = 'output/dev.parses.post'

def get_fallback_parses():
    fallback_parses = []
    with open(FALLBACK, 'r') as fallback_file:
        for line in fallback_file:
            fallback_parses.append(line)
    return fallback_parses

def preprocess(tree_filename, pre_filename):
    count = defaultdict(int)
    trees = []
    with open(tree_filename, 'r') as tree_file:
        for line in tree_file:
            t = Tree.from_str(line)
            t.binarize()
            t.remove_unit()
            for leaf in t.leaves():
                count[leaf.label] += 1
            trees.append(t)

    with open(pre_filename, 'w') as pre_file:
        for t in trees:
            for leaf in t.leaves():
                if count[leaf.label] < 2:
                    leaf.label = '<unk>'
            if MOD == 1:
                t.bigram()
            elif MOD == 2:
                t.trigram()
            elif MOD == 3:
                t.pair()
            elif MOD == 4:
                t.bigram()
                t.pair()
            pre_file.write(str(t)+'\n')

def postprocess(output_filename):
    fallback_parses = get_fallback_parses()
    with open(output_filename, 'r') as output_file:
        with open(output_filename+'.post', 'w') as output_file_post:
            l = 0
            for line in output_file:
                try:
                    t = Tree.from_str(line)
                    if MOD == 1:
                        t.unbigram()
                    elif MOD == 2:
                        t.untrigram()
                    elif MOD == 3:
                        t.unpair()
                    elif MOD == 4:
                        t.unpair()
                        t.unbigram()
                    t.restore_unit()
                    t.unbinarize()
                    output_file_post.write(str(t)+'\n')
                except:
                    output_file_post.write(fallback_parses[l])
                l += 1

if __name__ == '__main__':
    trees_file = 'trees/train.trees'
    pre_file = 'trees/train.trees.mod'
    strings_file = 'strings/dev.strings'
    output_file = 'output/dev.parses.mod'

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', action='store_true', help='print debugging statements')
    parser.add_argument('-m', help='choose the modification to use [1-3]')
    parser.add_argument('--trees', help='use the trees in the specified file')
    parser.add_argument('--pre', help='output file for pre-processed training data')
    parser.add_argument('--strings', help='use the strings in the specified file')
    parser.add_argument('--output', help='specify an output file')
    parser.add_argument('--fallback', help='file with the fallback string parses')
    args = parser.parse_args()

    DEBUG = args.d
    if args.trees:
        trees_file = args.trees
    if args.pre:
        pre_file = args.pre
    if args.strings:
        strings_file = args.strings
    if args.output:
        output_file = args.output
    if args.m:
        MOD = int(args.m)
    if args.fallback:
        FALLBACK = args.fallback

    preprocess(trees_file, pre_file)
    cfg = Grammar()
    cfg.construct_cfg(pre_file)
    cky = CKY(cfg)
    cky.parse_sentences(strings_file, output_file)
    postprocess(output_file)
