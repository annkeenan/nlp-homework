import lm
import ibm_model1
import fst
import math


# Process and store the data in a specified file
def process(filename):
    data = []
    # Read in the data
    with open(filename, 'r') as f:
        for line in f:
            data.append(line.rstrip().split())
    return data


# Translation model with transitions (fw, ew)
def make_tm(t, testfile):
    tm = fst.FST()
    tm.set_start('q0')
    tm.set_accept('q1')
    tm.add_transition(fst.Transition('q0', (fst.STOP, fst.STOP), 'q1'))
    known_words = set()
    for trans, prob in t.items():
        known_words.add(trans[0])
        if trans[1] == '∅':
            trans = (trans[0], fst.EPSILON)
        tm.add_transition(fst.Transition('q0', trans, 'q0'), prob)
    # Add unknown words from the test data
    with open(testfile) as f:
        prob = math.pow(10, -100)
        for line in f:
            for w in line.rstrip().split():
                if w not in known_words:
                    tm.add_transition(fst.Transition('q0', (w, fst.EPSILON), 'q0'), prob)
    return tm


# FST that maps a foreign sentence f to itself
def make_fm(f):
    fm = fst.FST()
    fm.set_start(0)
    for i, fw in enumerate(f):
        fm.add_transition(fst.Transition(i, (fw, fw), i+1))
    fm.add_transition(fst.Transition(
        len(fs), (fst.STOP, fst.STOP), len(fs)+1))
    fm.set_accept(len(f)+1)
    return fm


# Find the best path through FST m
def viterbi(m, key=None):
    vit = {}
    ptr = {}
    vit[m.start] = 0
    ptr[m.start] = None
    for q in sorted(m.states, key=key):
        if q == m.start:
            continue
        vit[q] = float("-inf")
        for t, wt in m.transitions_to[q].items():
            score = vit[t.q] + (math.log(wt) if wt > 0 else float("-inf"))
            if score > vit[q]:
                vit[q] = score
                ptr[q] = t
    path = []
    q = m.accept
    while q != m.start:
        if q not in ptr:
            raise ValueError("no path found")
        path.append(ptr[q])
        q = ptr[q].q
    path.reverse()
    return vit[m.accept], path


if __name__ == '__main__':
    print('========== CONSTRUCT MODELS ==========')
    mlm = lm.make_kneserney(process('data/train.en'), 2)
    mtm = make_tm(ibm_model1.make(), 'data/test.zh')
    # Iterate through each line of the data
    print('========== TESTING ===================')
    with open('data/test.mod2.out', 'w') as wf:
        for i, fs in enumerate(process('data/test.zh')):
            mf = make_fm(fs)
            # Compose all fst models
            m = fst.compose(fst.compose(mf, mtm), mlm)
            # Calculate the shortest path
            try:
                wt, path = viterbi(m, key=lambda q: (q[0][0], -1*len(q[1])))
                out = " ".join([t.a[1] for t in path[:-1] if t.a[1] != fst.EPSILON])
                # Print first 10 translations
                if i < 10:
                    print('%s' % out)
                # Print output to help with tracking how far along translations are
                # else:
                    # print('\rLine #: %d' % i, end='')
                wf.write('%s\n' % out)
            except ValueError as e:
                if i < 10:
                    print('')
                wf.write('\n')
