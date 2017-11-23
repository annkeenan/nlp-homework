import collections

class Fivegram(object):
    def __init__(self):
        self.counts = collections.Counter()
        self.prev_word = '<s>'*4
        self.total_count = 0.
        self.words = set()

    def train(self, filename):
        """Train the model on a text file."""
        for line in open(filename):
            for w in line.rstrip('\n'):
                self.read(w)

    def start(self):
        """Reset the state to the initial state."""
        self.prev_word = '<s>'*4

    def read(self, w):
        """Read in character w, updating the state."""
        self.words.add(w)

        self.counts[self.prev_word+w] += 1
        self.total_count += 1

        if self.prev_word[0:3] == '<s>':
            self.prev_word = self.prev_word[3:] + w
        else:
            self.prev_word = self.prev_word[1:] + w

    def prob(self, w):
        """Return the probability of the next character being w given the
        current state."""
        return self.counts[self.prev_word+w]/self.total_count

    def predict(self):
        max_w = ''
        max_w_prob = 0
        for w in self.words:
            if self.prob(w) > max_w_prob:
                max_w_prob = self.prob(w)
                max_w = w
        return max_w
