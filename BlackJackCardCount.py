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

    fig = plt.plot(midcard_percent, startValue, '-bo', [12/52, 12/52], [cvmin-.001, cvmax+.001], '--g')
    ax = plt.subplot(111)
    ax.set_xlim(.13, .35)
    xl = plt.xlabel('% Mid Cards in Deck')
    yl = plt.ylabel('Expected Game Value from Start State')
    xl.set_style('italic')
    yl.set_style('italic')
    ttl = plt.title('Shown Variability from Midcard Value given Count, Start State')
    ttl.set_weight('bold')
    ax.xaxis.set_tick_params(size=0)
    ax.yaxis.set_tick_params(size=0)
    ax.grid('on', linestyle='--')
    ax.legend(['Expected Value Given Midcard', 'Mean Midcard State'], loc=1)
    # plt.legend('Change in E related to midccard', 'Mean Midcard State')
    plt.savefig('PiCompCount.png', bbox_inches='tight')
    plt.clf()

    midcard_avg = float(12)
    pi_avg = cleanpolicy(read_obj('Midcard {} pi'. format(midcard_avg - 12)))
    policy_comparison = {}

    for midcard in midcards:
        pi = cleanpolicy(read_obj('Midcard {} pi'.format(midcard - 12)))
        similar = 0
        for state, action in pi.items():
            player, dealer, count = state
            if pi_avg[(player, dealer, count)] == action: similar += 1
        policy_comparison[midcard/52] = similar/len(pi)

    fig = plt.plot(policy_comparison.keys(), policy_comparison.values(), '-bo', [12/52, 12/52], [.94, 1.015], '--g')
    ax = plt.subplot(111)
    xl = plt.xlabel('% Mid Cards in Deck')
    yl = plt.ylabel('% Similar actions to typical deck')
    ttl = plt.title('Shown Variability of policy dependent on midcard')
    ttl.set_weight('bold')
    xl.set_style('italic')
    yl.set_style('italic')
    ax.xaxis.set_tick_params(size=0)
    ax.yaxis.set_tick_params(size=0)
    ax.grid('on', linestyle='--')
    ax.legend(['Normalized Player Utility', 'Mean Midcard State'], loc=1)
    plt.savefig('PiCompMid.png', bbox_inches='tight')
    plt.clf()

def cleanpolicy(policy):
    newpolicy = {}
    for key, value in policy.items():
        if value:
            newpolicy[key] = value
    return newpolicy

def main():
    '''
    Iterate through different card counts, find associated policies and expected values. Save to output file .pkl.

    :return: When called, this main script returns and saves policies and expected values associated with the value
    iteration algorithm for different card counts in blackjack. Files are of the format .pkl.
    '''

    mdp = BlackjackMDP()
    startState = mdp.startState()

    # Checking code for all succReward actions, can toggle on and off with booleans
    checkdouble = False
    checkbegin = False
    checkdraw = False
    checkstay = False
    count = 0
    check_succ(checkdouble, checkbegin, checkdraw, checkstay, count, mdp)

    # Generate data, store data for GUI. Generate data for different card counts found in list below.
    counts = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    countAnalysis = False
    if countAnalysis:
        for count in counts:
            countIteration(count, 12)

    pi_avg_count = cleanpolicy(read_obj('Count 0 Policy'))
    V_start_count = read_obj('Count 0 V')
    policy_comparison_count = {}
    startVCount = []
    for count in counts:
        pi_count = cleanpolicy(read_obj('Count {} Policy'.format(count)))
        print(pi_count)
        V_count = read_obj('Count {} V'.format(count))
        similar_action = 0
        for state, action in pi_count.items():
            player, dealer, count = state
            if pi_avg_count[(player, dealer, 0)] == action: similar_action += 1
        policy_comparison_count[count] = similar_action/len(pi_avg_count)
        startVCount.append(V_count[('', '', count)])

    fig = plt.plot(policy_comparison_count.keys(), policy_comparison_count.values(), '-bo', [0, 0], [.8, 1.025], '--k')
    ax = plt.subplot(111)
    xl = plt.xlabel('True Card Count')
    xl.set_style('italic')
    yl = plt.ylabel('% Similar actions to typical deck')
    yl.set_style('italic')
    ttl = plt.title('Difference in optimum policy over different Counts')
    ax.grid('on', linestyle='--')
    ttl.set_weight('bold')
    ax.xaxis.set_tick_params(size=0)
    ax.yaxis.set_tick_params(size=0)
    ax.legend(['% Similar to Mean Policy', 'Mean Policy Count State'], loc=1)
    plt.savefig('PolicyCompCount.png', bbox_inches='tight')
    plt.clf()

    fig = plt.plot(counts, startVCount, '-bo', [0, 0], [-.05, .05], '--k')
    ax = plt.subplot(111)
    yl = plt.ylabel('Normalized Player Expected Value')
    xl = plt.xlabel('True Card Count')
    xl.set_style('italic')
    yl.set_style('italic')
    ttl = plt.title('Expected Value given True Count')
    ttl.set_weight('bold')
    ax.xaxis.set_tick_params(size=0)
    ax.yaxis.set_tick_params(size=0)
    ax.grid('on', linestyle='--')
    ax.legend(['Normalized Player Utility', 'Mean Count State'], loc=1)
    plt.savefig('VCompCount.png', bbox_inches='tight')
    plt.clf()

    # Midcard analysis, how do value results change by entering midcard value
    midcards = [8+i/2 for i in range(0, 17)]
    midcardAnalysis = False
    if midcardAnalysis:
        for midcard in midcards:
            midcardIteration(midcard, 0)
    midcardPlot(midcards)

if __name__ == '__main__':
    main()