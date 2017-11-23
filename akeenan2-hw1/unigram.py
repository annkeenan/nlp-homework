import collections
import math

class Unigram(object):
    """Barebones example of a language model class."""

    def __init__(self):
        self.counts = collections.Counter()
        self.total_count = 0.

    def train(self, filename):
        """Train the model on a text file."""
        for line in open(filename):
            for w in line.rstrip('\n'):
                self.counts[w] += 1
                self.total_count += 1

    def start(self):
        """Reset the state to the initial state."""
        self.total_count = 0.

    def read(self, w):
        """Read in character w, updating the state."""
        self.counts[w] += 1
        self.total_count += 1

    def prob(self, w):
        """Return the probability of the next character being w given the
        current state."""
        return self.counts[w]/self.total_count

    def predict(self):
        return self.counts.most_common(1)[0][0]
