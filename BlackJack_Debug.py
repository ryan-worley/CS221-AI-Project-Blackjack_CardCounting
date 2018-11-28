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


def getCount(deck):
    return sum(deck[8:]) - sum(deck[:5])


def BlackJackFeatures(state, action):
    player, dealer, cards = state
    mdp = BlackjackMDP()

    if player != '':
        count = getCount(cards)
        value = mdp.cards_value(player)

    if dealer != '':
        dealercard = dealer[-1]


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

if __name__ == '__main__':
    main()