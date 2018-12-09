import random
import pickle
import collections
import Completed_FullBlackJack_Exact as ExactMDP



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
    if state == '' or state == 21:
        return state
    newState = list(state)
    if newState[-1].isdigit():
        return state + '*'
    for i, l in enumerate(state):
        if l.isalpha():
            newState.insert(i, '*')
            return ''.join(newState)

def fixDealerState(dealer, cards):
    if sum(cards[2]) == 0:
        assert ValueError
    if dealer == '11A':
        return 11
    elif dealer:
        return int(dealer)
    else:
        return dealer

# Perform |numTrials| of the following:
# On each trial, take the MDP |mdp| and an RLAlgorithm |rl| and simulates the
# RL algorithm according to the dynamics of the MDP.
# Each trial will run for at most |maxIterations|.
# Return the list of rewards that we get for each trial.
def simulate(mdp, rl, betPi, numTrials=1000, verbose=False, mincards=10, single_hand=False):
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
    for trial in range(numTrials):
        state = mdp.startState()
        statesequence = [state]
        totalReward = 0
        rewardsequence = []
        actionsequence = []
        handnum = 1
        while True:
            count = state[3]
            # Adjust true count so stays within limits
            if count > 10:
                count = 10
            elif count < -10:
                count = -10

            if state[1] is None:
                if sum(state[2]) < mincards:
                    break
                if single_hand = True:
                    break
                action = 'Begin'
                count = state[3]
                mdp.editBet(betPi[count])
                adjusted_state = ('', '', state[2], state[3])
                state = adjusted_state
            else:
                dealer = fixDealerState(state[1], state[2])
                action = rl.getAction((fixPlayerState(state[0]), dealer, count))

            # Choose random trans state
            transitions = mdp.succAndProbReward(state, action)
            i = sample([prob for newState, prob, reward in transitions])

            # Pull out state, prob, reward from transition
            newState, prob, reward = transitions[i]

            # Add to sequence of items
            actionsequence.append(action)
            rewardsequence.append(reward)
            statesequence.append(newState)
            totalReward += reward
            state = newState
        if verbose:
            print("Trial %d (totalReward = %s): %s" % (action, reward, newState))
        totalRewards.append(totalReward)
        totalStates.append(statesequence)
        totalActions.append(actionsequence)
    print('Total Average Reward is: {}'. format(sum(totalRewards)))
    return totalRewards, totalStates, totalActions

def loadPolicy():
    policy = collections.defaultdict(str)
    counts = [i for i in range(-10, 11)]
    for count in counts:
        current_pi = pickle.load(open('./policy/' + 'Count {} Policy.pkl'.format(count), 'rb'))
        policy.update(current_pi)
    return policy

def main():
    pi = loadPolicy()
    rl = FixedRLAlgorithm(pi)
    trueMDP = ExactMDP.BlackjackMDP()

    bet_test=True
    if bet_test:
        betpi = betpolicy(0, 2, .5, 1, 5)
        simulate(mdp=trueMDP, rl=rl, betPi=betpi)
    if policy_test:
        betpi = betpolicy(0, 2, 1, 1, 1)
        mincards =



if __name__ == '__main__':
    main()
