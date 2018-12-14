import random
import pickle
import collections
import Completed_FullBlackJack_Exact as ExactMDP
import numpy as np
import scipy.io as sio

# Abstract class: an RLAlgorithm performs reinforcement learning.  All it needs
# to know is the set of available actions to take.  The simulator (see
# simulate()) will call getAction() to get an action, perform the action, and
# then provide feedback (via incorporateFeedback()) to the RL algorithm, so it can adjust
# its parameters.
class RLAlgorithm:
    # Your algorithm will be asked to produce an action given a state.
    def getAction(self, state): raise NotImplementedError("Override me")

    # We will call this function when simulating an MDP, and you should update
    # parameters.
    # If |state| is a terminal state, this function will be called with (s, a,
    # 0, None). When this function is called, it indicates that taking action
    # |action| in state |state| resulted in reward |reward| and a transition to state
    # |newState|.
    def incorporateFeedback(self, state, action, reward, newState): raise NotImplementedError("Override me")

# An RL algorithm that acts according to a fixed policy |pi| and doesn't
# actually do any learning.
class FixedRLAlgorithm(RLAlgorithm):
    def __init__(self, pi): self.pi = pi

    # Just return the action given by the policy.
    def getAction(self, state): return self.pi[state]

    # Don't do anything: just stare off into space.
    def incorporateFeedback(self, state, action, reward, newState): pass

def betpolicy(lower, upper, loweramt, midamt, upperamt):
    policy = collections.defaultdict(float)
    for i in range(-10, 11):
        if i < lower:
            policy[i] = loweramt
        elif lower <= i <= upper:
            policy[i] = midamt
        else:
            policy[i] = upperamt
    return policy

def fixPlayerState(state):
    if state == '':
        return state
    if state == '21AD':
        return 21
    newState = list(state)
    if newState[-1].isdigit():
        return state + '*'
    for i, l in enumerate(state):
        if l.isalpha():
            newState.insert(i, '*')
            return ''.join(newState)

def fixDealerState(dealer, cards):
    if sum(cards) == 0:
        assert ValueError
    if dealer == '11A':
        return 11
    elif dealer:
        return int(dealer)
    else:
        return dealer

def adjustCount(count):
    if count > 10:
        return 10
    elif count < -10:
        return -10
    else:
        return count

# Perform |numTrials| of the following:
# On each trial, take the MDP |mdp| and an RLAlgorithm |rl| and simulates the
# RL algorithm according to the dynamics of the MDP.
# Each trial will run for at most |maxIterations|.
# Return the list of rewards that we get for each trial.
def simulate(mdp, rl, betPi, numTrials=1000, verbose=False, mincards=15, single_hand=False):
    '''Simulates a game of Blackjack based on a fixed policy entered in by the user of the function
    and based on the mechanics of an MDP. A betting strategy must be implemented as well for testing.

    mincards = minimum number of cards left in the deck where a new hand is allowed to be started.
    verbose = displays lots of information
    single_hand = if you only want to run an analysis for one hand toggle this option on
    numTrials = number of Monte Carlo trials conducted
    '''
    def sample(probs):
        target = random.random()
        accum = 0
        for i, prob in enumerate(probs):
            accum += prob
            if accum >= target: return i
        raise Exception("Invalid probs: %s" % probs)


    totalRewards = []  # The rewards we get on each trial
    totalStates = []
    totalActions = []
    count_counts = []
    for trial in range(numTrials):
        state = mdp.startState()
        statesequence = [state]
        totalReward = 0
        rewardsequence = []
        actionsequence = []
        handreward = []
        hands = []
        handnum = 1
        while True:
            count = state[3]
            # Adjust true count so stays within limits
            count = adjustCount(count)

            if state[1] is None:
                if sum(state[2]) < mincards or single_hand:
                    break
                handnum += 1
                action = 'Begin'
                mdp.editBet(betPi[count])
                state = ('', '', state[2], state[3])
                count_counts.append(state[3])
            else:
                action = rl.getAction((fixPlayerState(state[0]), fixDealerState(state[1], state[2]), count))

            # Choose random trans state
            transitions = mdp.succAndProbReward(state, action)
            i = sample([prob for _, prob, _ in transitions])

            # Pull out state, prob, reward from transition
            newState, prob, reward = transitions[i]

            # Add to sequence of items
            actionsequence.append(action)
            rewardsequence.append(reward)
            statesequence.append(newState)
            totalReward += reward
            state = newState

        if verbose:
            print("Trial {}, totalReward = {}, number of hands = {}, handReward = {}".format(trial, totalReward, handnum, totalReward/handnum))

        hands.append(handnum)
        totalRewards.append(totalReward)
        handreward.append(totalReward/handnum)
        totalStates.append(statesequence)
        totalActions.append(actionsequence)

    print('Total Average Reward is: {}'. format(sum(totalRewards)/len(totalRewards)))
    print('Total Average Hand Reward is: {}'. format(sum(handreward)/len(handreward)))
    return totalRewards, totalStates, totalActions, hands, sum(totalRewards)/len(totalRewards), sum(handreward)/len(handreward), count_counts

def loadPolicy():
    '''Load all count policy files and put them into one dict'''
    policy = collections.defaultdict(str)
    counts = [i for i in range(-15, 16)]
    for count in counts:
        current_pi = pickle.load(open('./policy/' + 'Count {} Policy.pkl'.format(count), 'rb'))
        policy.update(current_pi)
    return policy

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

def read_bet(name):
    """
    Summary: Reads .pkl file with specified name and directly path
    :param name: Directory to access file
    :return: Returns the .pkl file stored as a single variable
    """
    with open('./policy/' + name + '.pkl', 'rb') as f:
        obj = pickle.load(f)
        f.close()
        return obj


def bet_testing(rl, trueMDP, betpolicies):
    num = 1
    '''Load necessary bet data into matlab as a count distribution'''
    for bet in betpolicies:
        print('Bet policy {}'.format(num))
        betpi = betpolicy(bet[0], bet[1], bet[2], bet[3], bet[4])
        tr, ts, ta, h, avgV, avgVHand, count_count = simulate(mdp=trueMDP, rl=rl, betPi=betpi, numTrials=2500, verbose=True)
        save_obj('{}_{}_{}_BetResults'.format(bet[2], bet[3], bet[4]), [tr, ts, ta, h, avgV, avgVHand, count_count])
        num += 1
        sio.savemat('CountDistribution', {'counts': count_count})

def main():
    '''Main file, no inputs, conducts simulation of different betting polciies'''
    # Initialize Objects for testing
    pi = loadPolicy()
    rl = FixedRLAlgorithm(pi)
    trueMDP = ExactMDP.BlackjackMDP(multiplicity=8)
    betpolicies = ([0, 2, .5, 1, 5],
                   [0, 2, 0, 1, 5],
                   [0, 2, 0, 0, 5],
                   [0, 2, 1, 1, 1],
                   [0, 2, .5, 1, 2.5])
    '''Bet Testing Section'''
    # Toggle Bet test analysis
    bet_test = False
    if bet_test:
        bet_testing(rl, trueMDP, betpolicies)

    '''Policy Testing section'''
    # Toggle Policy Testing Section
    policy_test = False
    if policy_test:
        tr, ts, ta, h, avgV, avgVHand, count_count = simulate(mdp=trueMDP, rl=rl, betPi=betpolicies[4], numTrials=1000, verbose=True)

    '''Analyzing Bet Strategy Section'''
    # Toggle Analyzing
    Analyzing = True
    if Analyzing:
        avg_Reward = []
        handReward = []
        V_betSimulated = []
        for i, bet in enumerate(betpolicies):
            data = read_bet('{}_{}_{}_BetResults'.format(bet[2], bet[3], bet[4]))
            totalReward, totalStates, totalActions, totalHands, avgV, avgVHand, counts_count = data
            print(np.mean(totalHands))
            print('Policy {} '.format(i+1), avgV/(np.mean(totalHands)))
            avg_Reward.append(sum(totalReward)/len(totalReward))
            for tr, th in zip(totalReward, totalHands):
                handReward.append(tr/th)
            V_betSimulated.append(sum(handReward)/len(handReward))

        file = sio.loadmat('EV_Exact.mat')
        V_betExact = file['EV']
        print(V_betExact)

        V_Compare = [(x, y) for x, y in zip(V_betSimulated, V_betExact)]
        V_Difference = [x -y for x, y in zip(V_betSimulated, V_betExact)]
        V_percentDifference = [((x - y)/x)*100 for x, y in zip(V_betSimulated, V_betExact)]
        print('Hand Values for 5 cases ', V_Compare)
        print('Difference for Each Case ', V_Difference)
        print('Normalized Percent Different ', V_percentDifference)


    '''Create action log from ran data, not used just interesting'''
    actions = collections.defaultdict(int)
    # Create Action Log
    actionAnalysis = False
    if actionAnalysis:
        for action_sequence in ta:
            for action in action_sequence:
                actions[action] += 1
    countAnalysis = False
    if countAnalysis:
        count = collections.defaultdict(int)
        counts = []
        for state_seq in ts:
            for state in state_seq:
                count[state[3]] += 1
                counts.append(state[3])
        print(actions, count)
        mean = np.mean(counts)
        std = np.std(counts, ddof=1)
        print(counts)
        print(mean, std)


if __name__ == '__main__':
    main()
