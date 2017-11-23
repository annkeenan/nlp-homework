import sys
import cer
import fst
import viterbi
import fst_wrapper

class FSTM_Decode:
    def __init__(self):
        self.old_train = list(open("data/train.old"))
        self.new_train = list(open("data/train.new"))
        self.old_test = list(open("data/test.old"))
        self.new_test = list(open("data/test.new"))
        self.num_lines = len(self.old_test) # used for the status bar
        self.fst_m = fst.FST() # composition of lm and tm

    def status_bar(self, completed):
        fraction = completed/self.num_lines
        i = int(fraction * 50)
        print("\r[%-50s] %d%%" % ('='*i, fraction*100), end='')

    def train(self):
        fst_mlm = fst.make_ngram(self.new_train, 2)
        fst_mtm = fst_wrapper.get_fst_mtm(self.old_train, self.new_train)
        self.fst_m = fst.compose(fst_mlm, fst_mtm)

    # convert the old line to modern text
    def predict(self, old_line, print_lines=True):
        fst_mw = fst_wrapper.get_fst_mw(old_line)
        _fst = fst.compose(self.fst_m, fst_mw)
        path = viterbi.viterbi_path(_fst)
        # reconstruct the path
        predicted_line = ''
        for p in path[0]:
            if p[0][0][0] in _fst.input_alphabet:
                predicted_line += p[0][0][0]
        # print out the modern line with log probability
        if print_lines:
            print(predicted_line, end='')
            print(path[1])
        return predicted_line

    def test(self, lines=10):
        # test first 10 lines
        for line in self.old_test[:lines]:
            self.predict(line)

    def test_all(self, output_file):
        # test all lines of test file
        l = 0
        self.status_bar(l)
        with open(output_file, 'w') as f:
            for line in self.old_test:
                f.write(self.predict(line, False))
                l += 1
                self.status_bar(l)
        print()
        print(cer.cer(zip(self.new_test, list(open(output_file)))))
