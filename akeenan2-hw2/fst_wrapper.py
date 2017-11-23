import fst
import random
import sys

input_alphabet = set()
output_alphabet = set()

# typo model
def get_fst_mtm(old_data, new_data, initialize=True):
    m = fst.FST()
    m.set_start("q0")
    # get the old and modern alphabets from the traning data
    for old_line in old_data:
        for w in old_line:
            output_alphabet.add(w)
    for new_line in new_data:
        for w in new_line:
            input_alphabet.add(w)
    # generate the typo model
    for output_w in output_alphabet:
        m.add_transition(fst.Transition("q0", (fst.EPSILON, output_w), "q0")) #insert
    for input_w in input_alphabet:
        m.add_transition(fst.Transition("q0", (input_w, fst.EPSILON), "q1")) #delete
        for output_w in output_alphabet: # substitute
            m.add_transition(fst.Transition("q1", (input_w, output_w), "q0"))
            m.add_transition(fst.Transition("q0", (input_w, output_w), "q0"))
    # add terminal transitions
    m.add_transition(fst.Transition("q0", (fst.STOP, fst.STOP), "q2"))
    m.add_transition(fst.Transition("q1", (fst.STOP, fst.STOP), "q2"))
    m.set_accept("q2")
    # initialize the weights
    if initialize:
        for state in m.states:
            for transition in m.transitions_from[state].keys():
                # higher probability if going to the same character
                if transition.a[0] == transition.a[1]:
                    m.reweight_transition(transition, 100)
                else:
                    m.reweight_transition(transition, 1)
    m.normalize_cond()
    return m

# specific to a word/line
def get_fst_mw(word):
    m = fst.FST()
    m.set_start("q0")
    n = 1
    for w in word:
        m.add_transition(fst.Transition("q"+str(n-1), (w, w), "q"+str(n)))
        n += 1
    m.add_transition(fst.Transition("q"+str(n-1), (fst.STOP, fst.STOP), "q"+str(n)))
    m.set_accept("q"+str(n))
    return m
