import copy
import sys
import collections
from datetime import datetime
import cProfile
import re
import pickle
import itertools
from ValueIteration import *
from BlackJackMDP import *


def save_obj(name, obj):
    with open('policy/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, 0)
        f.close()


def read_obj(name):
    with open('policy/' + name + '.pkl', 'rb') as f:
        obj = pickle.load(f)
        f.close()
        return obj

def main():
    mdp = BlackjackMDP()
    checkdouble = True
    checkbegin = False
    checkdraw = False
    checkstay = False
    count = 0

    if checkdouble:
        result = mdp.succAndProbReward(('12*D', 9, 0), 'Double')
        print('Check Double', result, sum(x[1] for x in result))

    if checkbegin:
        result = mdp.succAndProbReward(('', '', count), 'Begin')
        print('Check Begin', result, sum(x[1] for x in result))

    if checkdraw:
        result = mdp.succAndProbReward(('12*A', '4', count), 'Draw')
        print('Check Draw', result, sum(x[1] for x in result))

    if checkstay:
        result = mdp.succAndProbReward(('17', '12', count), 'Stay')
        print('Check Stay', result, sum(x[1] for x in result))

    counts = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for count in counts:
        mdp = BlackjackMDP(count=count, midcards=12)
        alg = ValueIteration()
        alg.solve(mdp, .001)
    #     startState = mdp.startState()
    #     print('Expected Value for count {}: {}'.format(count, alg.V[startState]))
    #     print('Algorithm Value iteration with count {}'.format(count))
    #     save_obj('Count {} Policy'.format(count), alg.pi)
    #     print('Saved policy for count {}'.format(count))
    #     save_obj('Count {} V'.format(count), alg.V)
    #     print('Saved Expected Value Value for count {}'.format(count))

if __name__ == '__main__':
    main()