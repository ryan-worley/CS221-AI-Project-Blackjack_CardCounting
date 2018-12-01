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
import matplotlib.pyplot as plt


def save_obj(name, obj):
    """
    Summary: Save object as a pickle file to associated input directory and filename
    :param name: File name and directory for .pkl output file
    :param obj: Variable, object you wish to save
    :return: No returns, file is just saved
    """
    with open('policy/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, 0)
        f.close()


def read_obj(name):
    """
    Summary: Reads .pkl file with specified name and directly path
    :param name: Directory to access file
    :return: Returns the .pkl file stored as a single variable
    """
    with open('./policy/' + name + '.pkl', 'rb') as f:
        obj = pickle.load(f)
        f.close()
        return obj


def check_succ(checkdouble, checkbegin, checkdraw, checkstay, count, mdp):
    if checkdouble:
        result = mdp.succAndProbReward(('12*D', 9, mdp.cardcount), 'Double')
        print('Check Double', result, sum(x[1] for x in result))

    if checkbegin:
        result = mdp.succAndProbReward(('', '', mdp.cardcount), 'Begin')
        print('Check Begin', result, sum(x[1] for x in result))

    if checkdraw:
        result = mdp.succAndProbReward(('12*A', '4', mdp.cardcount), 'Draw')
        print('Check Draw', result, sum(x[1] for x in result))

    if checkstay:
        result = mdp.succAndProbReward(('17', '12', mdp.cardcount), 'Stay')
        print('Check Stay', result, sum(x[1] for x in result))

def countIteration(count, midcards):
    mdp = BlackjackMDP(count=count, midcards=12)
    alg = ValueIteration()
    alg.solve(mdp, .001)
    startState = mdp.startState()
    print('Expected Value for count {}: {}'.format(count, alg.V[startState]))
    print('Algorithm Value iteration with count {}'.format(count))
    save_obj('Count {} Policy'.format(count), alg.pi)
    print('Saved policy for count {}'.format(count))
    save_obj('Count {} V'.format(count), alg.V)
    print('Saved Expected Value Value for count {}'.format(count))

def midcardIteration(midcard, count):
    mdp = BlackjackMDP(count=0, midcards=midcard)
    alg = ValueIteration()
    alg.solve(mdp, .001)
    startState = mdp.startState()
    save_obj('Midcard {} V'.format(midcard - 12), alg.V)
    print('Saved Expected Value for midcard {}'.format(midcard))
    save_obj('Midcard {} pi'.format(midcard - 12), alg.pi)
    print('Saved Expected Value for midcard {}'.format(midcard))

def midcardPlot(midcards):
    # Plotting for midcard analysis
    Vs = {}
    mdp = BlackjackMDP(count=0, midcards=12)
    startState = mdp.startState()
    startValue = []
    drawValue = []

    for midcard in midcards:
        V = read_obj('Midcard {} V'.format(midcard-12))
        Vs[midcard] = V
        startValue.append(V[startState])

    midcard_percent = [m/52 for m in midcards]
    cvmin = min(startValue)
    cvmax = max(startValue)

    plt.plot(midcard_percent, startValue, '-bo', [12/52, 12/52], [cvmin-.001, cvmax+.001], '--g')
    plt.ylabel('% Mid Cards in Deck')
    plt.xlabel('Expected Game Value from Start State')
    plt.title('Shown Variability from Midcard Value given Count, Start State')
    # plt.legend('Change in E related to midccard', 'Mean Midcard State')
    plt.show()


def main():
    '''
    Iterate through different card counts, find associated policies and expected values. Save to output file .pkl.

    :return: When called, this main script returns and saves policies and expected values associated with the value
    iteration algorithm for different card counts in blackjack. Files are of the format .pkl.
    '''

    mdp = BlackjackMDP()

    # Checking code for all succReward actions, can toggle on and off with booleans
    checkdouble = False
    checkbegin = False
    checkdraw = False
    checkstay = False
    count = 0
    check_succ(checkdouble, checkbegin, checkdraw, checkstay, count, mdp)

    # Generate data, store data for GUI. Generate data for different card counts found in list below.
    counts = [-10, -9, -8, -7, -6, -5, -4, 6, 7, 8, 9, 10]
    countAnalysis = False
    if countAnalysis:
        for count in counts:
            countIteration(count, 12)

    pi_avg_count = read_obj('Count 0 Policy')
    V_start_count = read_obj('Count 0 V')
    policy_comparison_count = {}
    startVCount = []
    for count in counts:
        pi_count = read_obj('Count {} Policy'.format(count))
        V_count = read_obj('Count {} Policy'.format(count))
        similar_action = 0
        for state, action in pi_count.items():
            if pi_avg_count[state] == action: similar_action += 1
        policy_comparison_count[count] = similar_action/len(pi_avg_count)
        startVCount.append(V_count[mdp.startState()])

    plt.plot(policy_comparison_count.keys(), policy_comparison_count.values(), '-bo', [0, 0], [.8, 1], '--g')
    plt.ylabel('Count of Deck')
    plt.xlabel('% Actions Similar to Count 0 Deck')
    plt.title('Difference in optimum policy over different Counts')
    plt.show()

    plt.plot(count, startVCount, '-bo', [12 / 52, 12 / 52], [-.05, .05], '--g')
    plt.ylabel('Normalized Player Expected Value')
    plt.xlabel('True Card Count')
    plt.title('Expected Value given True Count')
    plt.show()

    # Midcard analysis, how do value results change by entering midcard value
    midcards = [8+i/2 for i in range(0, 17)]
    midcardAnalysis = True
    if midcardAnalysis:
        for midcard in midcards:
            midcardIteration(midcard, 0)
    midcardPlot(midcards)

    midcard_avg = 12
    pi_avg = read_obj('Midcard {} pi'. format(midcard_avg - 12))
    policy_comparison = {}

    for midcard in midcards:
        pi = read_obj('Midcard {} pi'.format(midcard - 12))
        similar = 0
        for state, action in pi.items():
            if pi_avg[state] == action: similar += 1
        policy_comparison[midcard/52] = similar/len(pi)

    plt.plot(policy_comparison.keys(), policy_comparison.values(), '-bo', [12/52, 12/52], [.9, 1], '--g')
    plt.ylabel('% Mid Cards in Deck')
    plt.xlabel('% Similar actions to typical deck')
    plt.title('Shown Variability of policy dependent on midcard')
    plt.show()

if __name__ == '__main__':
    main()