import fst
import fst_wrapper
from collections import deque, Counter
import math

counts = Counter()

"""
Base code/algorithm for topological sort found at
https://algocoding.wordpress.com/2015/04/05/topological-sorting-python/
"""
def kahn_topsort(graph):
    in_degree = { state : 0 for state in graph.states }
    # initialize the in degree to number of transitions into each state
    for state in graph.states:
        for transition in graph.transitions_to[state].keys():
            in_degree[state] += 1
    # initialize the frontier
    Q = []
    for state in in_degree:
        if in_degree[state] == 0:
            Q.append(state)
    # add states from the frontier
    sorted_states = []
    while Q:
        state = Q.pop()
        sorted_states.append(state)
        for transition in graph.transitions_from[state].keys():
            in_degree[transition.r] -= 1
            if in_degree[transition.r] == 0:
                Q.append(transition.r)
    # if topological sort worked, return the list of sorted states
    if len(sorted_states) == len(graph.states):
        return sorted_states
    else:
        return []

# find the best path through the fst
def viterbi_path(fst, get_counts=False):
    viterbi = dict()
    pointer = dict()
    # topological soft
    sorted_states = kahn_topsort(fst)
    for state in sorted_states:
        if state == fst.start:
            viterbi[state] = 1
        else:
            viterbi[state] = 0
    # construct the best path transitions
    for state in sorted_states:
        for transition, weight in fst.transitions_to[state].items():
            if viterbi[transition.q] * weight > viterbi[state]:
                viterbi[state] = viterbi[transition.q] * weight
                pointer[state] = transition.q
    # reconstruct the path
    path = deque()
    state = fst.accept
    total_weight = 0.
    while state != fst.start:
        for transition, weight in fst.transitions_to[state].items():
            if state not in pointer: # error constructing path
                return None
            if transition.q == pointer[state]:
                # track the log probability of the path
                total_weight += math.log(weight)
                # if training
                if get_counts:
                    t = transition.composed_from[0].composed_from[1]
                    counts[t]+=1
                break
        path.appendleft(state)
        state = pointer[state]
    path.appendleft(state) # add start state
    return (path, total_weight)
