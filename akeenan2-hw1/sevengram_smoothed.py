import collections

class Sevengram_Smoothed(object):
    def __init__(self):
        self.counts = {}
        self.prev_word = {}
        for i in range(1,7):
            self.counts[i] = {}
            self.prev_word[i] = '<s>'*i
        self.total_count = 0.
        self.words = collections.Counter()
        self.d = {}
        self.c_u = collections.Counter()
        self.n_u = collections.Counter()

    def train(self, filename):
        """Train the model on a text file."""
        for line in open(filename):
            for w in line.rstrip('\n'):
                self.read(w)

        self.set_constants()

    def set_constants(self):
        one_count = {}
        two_count = {}
        for i in range(1,7):
            one_count[i] = 0
            two_count[i] = 0
            for _, prev_word in self.counts[i].items():
                for _, count in prev_word.items():
                    if count == 1:
                        one_count[i] += 1
                    elif count == 2:
                        two_count[i] += 1
            self.d[i] = one_count[i]/(one_count[i]+2*two_count[i])

    def set_state_constants(self):
        for i in range(1,7):
            if self.prev_word[i] in self.counts[i]:
                for _, count in self.counts[i][self.prev_word[i]].items():
                    self.c_u[i] += count
                self.n_u[i] = len(self.counts[i][self.prev_word[i]])

    def start(self):
        """Reset the state to the initial state."""
        for i in range(7):
            self.prev_word[i] = '<s>'*i

    def read(self, w):
        """Read in character w, updating the state."""
        self.words[w] += 1
        for i in range(1,7):
            if self.prev_word[i] not in self.counts[i]:
                self.counts[i][self.prev_word[i]] = collections.Counter()
            self.counts[i][self.prev_word[i]][w] += 1
            if self.prev_word[i][0:3] == '<s>':
                self.prev_word[i] = self.prev_word[i][3:] + w
            else:
                self.prev_word[i] = self.prev_word[i][1:] + w
        self.total_count += 1

    def prob(self, w):
        """Return the probability of the next character being w given the
        current state."""
        self.c_uw = {}
        for i in range(1,7):
            if self.prev_word[i] in self.counts[i]:
                self.c_uw[i] = self.counts[i][self.prev_word[i]][w]
            else:
                self.c_uw[i] = 0

        prob = {}
        prob[0] = max(0, self.words[w])/len(self.words)
        for i in range(1,7):
            if self.c_u[i] > 0:
                prob[i] = max(0, self.c_uw[i]-self.d[i])/self.c_u[i] + self.n_u[i]*self.d[i]/self.c_u[i] * prob[i-1]
            else:
                prob[i] = 0
        return prob[6]

    def predict(self):
        max_w = ''
        max_w_prob = 0
        self.set_state_constants()
        for w in self.words:
            if self.prob(w) > max_w_prob:
                max_w_prob = self.prob(w)
                max_w = w
        return max_w
