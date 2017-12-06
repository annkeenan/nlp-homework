from collections import defaultdict
import math


# Process and store the data in a specified file
def process(filename):
    data = []
    keys = set()
    # Read in the data
    with open(filename, 'r') as f:
        for line in f:
            l = line.rstrip().split('\t', 2)
            # Store the sentences locally
            fs = list(filter(None, l[0].split(' ')))
            es = list(filter(None, ['∅'] + l[1].split(' ')))
            data.append((fs, es))
            # Track the number of unique foreign keys
            for fw in fs:
                for ew in es:
                    keys.add((fw, ew))
    return data, len(keys)


# Calculate the log likelihood of the model
def _log_likelihood(data, t):
    ll = 0
    for (fs, es) in data:
        prodt = 1
        for fw in fs:
            sumt = 0
            for ew in es:
                sumt += t[(fw, ew)]
            prodt *= 1/(len(es)) * sumt
        P = 1/100 * prodt
        ll += math.log(P)
    return ll


# Convert to uniform probability
def _uniform(value):
    return lambda: value


def train(data, keys, iterations=10):
    # Initialize to uniform probability
    t = defaultdict(_uniform(float(1/keys)))
    ll = -math.inf
    i = 0
    while i < iterations:
        count = defaultdict(float)
        c_total = defaultdict(float)
        t_total = defaultdict(float)
        for (fs, es) in data:
            # E step
            for fw in fs:
                t_total[fw] = 0
                for ew in es:
                    t_total[fw] += t[(fw, ew)]
            for ew in es:
                for fw in fs:
                    count[(fw, ew)] += t[(fw, ew)] / t_total[fw]
                    c_total[ew] += t[(fw, ew)] / t_total[fw]
        # M step
        for (fw, ew) in count.keys():
            t[(fw, ew)] = count[(fw, ew)] / c_total[ew]
        # Calculate and print the log likelihood
        ll = _log_likelihood(data, t)
        print(ll)
        i += 1
    return t


def printp(t):
    pairs = [
        ('绝地', 'jedi'),
        ('机械人', 'droid'),
        ('原力', 'force'),
        ('原虫', 'midi-chlorians'),
        ('你', 'yousa')]
    for (fw, ew) in pairs:
        print('t(%s, %s) = %f' % (fw, ew, t[(fw, ew)]))


def test(data, t):
    # Write data to output
    with open('data/train.zh-en'+'.out', 'w') as wf:
        for (fs, es) in data:
            # Find the best alignment
            for fpos, fw in enumerate(fs):
                maxw = (-math.inf, [])  # (position, probability)
                for epos, ew in enumerate(es[1:]):
                    if ew != '∅':
                        if t[(fw, ew)] > maxw[0]:
                            maxw = (t[(fw, ew)], [epos])
                        elif t[(fw, ew)] == maxw[0]:
                            maxw[1].append(epos)
                for epos in maxw[1]:
                    wf.write('%d-%d ' % (fpos, epos))
            wf.write('\n')


def make():
    data, keys = process('data/train.zh-en')
    return train(data, keys)


if __name__ == '__main__':
    data, keys = process('data/train.zh-en')
    print('========== TRAINING ==================')
    t = train(data, keys)
    print('========== LOG PROBABILITIES =========')
    printp(t)
    print('========== TESTING ===================')
    test(data, t)
