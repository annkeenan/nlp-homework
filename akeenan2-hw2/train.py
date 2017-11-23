import sys
import cer
import fst
import viterbi
import fst_wrapper

class FSTM_Train:
    def __init__(self):
        self.old_train = list(open("data/train.old"))
        self.new_train = list(open("data/train.new"))
        self.old_test = list(open("data/test.old"))
        self.new_test = list(open("data/test.new"))
        self.fst_mtm = fst.FST() # typo model
        self.fst_mlm = fst.make_ngram(self.new_train, 2) # language model
        self.num_lines = 0 # used for the status bar

    def status_bar(self, completed):
        fraction = completed/self.num_lines
        i = int(fraction * 50)
        sys.stdout.write("\r[%-50s] %d%%" % ('='*i, fraction*100))
        sys.stdout.flush()

    def train(self, output_file, iterations):
        # construct the initial unweighted typo model
        self.fst_mtm = fst_wrapper.get_fst_mtm(self.old_train, self.new_train, False)
        for _ in range(iterations): # iterate
            self.num_lines = len(self.old_train) # for the status bar
            l = 0
            self.status_bar(l) # track progress with a status bar
            # train on parallel text
            for old_line, new_line in zip(self.old_train, self.new_train):
                # construct the fst models for modern and old lines
                fst_mm = fst_wrapper.get_fst_mw(new_line)
                fst_me = fst_wrapper.get_fst_mw(old_line)
                #compose the models and find the shortest path
                _fst = fst.compose(fst.compose(fst_mm, self.fst_mtm), fst_me)
                viterbi.viterbi_path(fst=_fst, get_counts=True)
                # reweight the tm with the new counts and reweight
                for t, count in viterbi.counts.items():
                    self.fst_mtm.reweight_transition(t, count)
                self.fst_mtm.normalize_cond(.01)
                l += 1
                self.status_bar(l)
            print() # add a line after the status bar
            self.predict(output_file)
            # print the overall score
            print('SCORE: ', end='')
            print(cer.cer(zip(self.new_test, list(open(output_file)))))

    def predict(self, output_file):
        self.num_lines = len(self.old_test) # for the status bar
        with open(output_file, 'w') as f:
            fst_m = fst.compose(self.fst_mlm, self.fst_mtm)
            l = 0
            for old_line, new_line in zip(self.old_test, self.new_test):
                fst_mw = fst_wrapper.get_fst_mw(old_line)
                # compose the lm, tm, and wm, and find the best path
                _fst = fst.compose(fst_m, fst_mw)
                path = viterbi.viterbi_path(_fst)
                predicted_line = ''
                for p in path[0]:
                    if p[0][0][0] in self.fst_mlm.input_alphabet:
                        predicted_line += p[0][0][0]
                # print out the first 10 lines
                if l < 10:
                    sys.stdout.write(predicted_line)
                    print(path[1])
                else: # show the progress bar
                    self.status_bar(l)
                l += 1
                # write the prediction to the file
                f.write(predicted_line)
        # print out the transitions with weight greater than 0.1
        for state, transitions in self.fst_mtm.transitions_to.items():
            for transition, weight in transitions.items():
                if weight >= 0.1:
                    print(str(transition) + ' = ' + str(weight))

    def test(self, output_file):
        # train 3 iterations
        self.train(output_file, 3)
