import collections

class Bigram(object):
    def __init__(self):
        self.charmap = dict()
        self.prev_w = '<s>'
        self.counts = collections.defaultdict(collections.Counter)
        self.total_count = 0.

    def start(self):
        self.prev_w = '<s>'

    def read_charmap(self, filename):
        for line in open(filename):
            l = line.split()
            self.charmap[l[0]] = l[1]

    def train(self, filename):
        for line in open(filename):
            for w in line.rstrip('\n'):
                self.read(w)

    def read(self, w):
        self.counts[w][self.prev_w] += 1
        self.prev_w = w
        self.total_count += 1

    def candidates(self, token):
        if token == '<space>':
            return [' ']
        elif len(token) == 1:
            possibilities = [token]
        else:
            possibilities = []
        for han, pin in self.charmap.items():
            if pin == token:
                possibilities.append(han)
        return possibilities

    def predict(self, token):
        candidates = self.candidates(token)
        if not candidates: # potentially a new word that doesn't exist in the dictionary
            return token
        max_count = 0
        max_prob_c = ''
        for candidate in candidates:
            count = self.counts[candidate].get(self.prev_w, 0)
            if count > max_count:
                max_count = count
                max_prob_c = candidate
            elif count == max_count:
                count = sum(self.counts[candidate].values())
                if count > sum(self.counts[max_prob_c].values()):
                    max_prob_c = candidate
        return max_prob_c
