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

# Perform |numTrials| of the following:
# On each trial, take the MDP |mdp| and an RLAlgorithm |rl| and simulates the
# RL algorithm according to the dynamics of the MDP.
# Each trial will run for at most |maxIterations|.
# Return the list of rewards that we get for each trial.
def simulate(mdp, rl, numTrials=1, maxGames=50, verbose=False,
             , betPi):
    # Return i in [0, ..., len(probs)-1] with probability probs[i].
    def sample(probs):
        target = random.random()
        accum = 0
        for i, prob in enumerate(probs):
            accum += prob
            if accum >= target: return i
        raise Exception("Invalid probs: %s" % probs)


    totalRewards = []  # The rewards we get on each trial
    statesequence = []
    for trial in range(numTrials):
        state = mdp.startState()
        statesequence = [state]
        totalReward = 0
        rewardsequence = []
        actionsequence = []
        for _ in range(maxGames):
            count = state[3]
            if count > 10:
                count = 10
            elif count < -10:
                count = -10

            if state[1] == None:
                if sum(state[2]) < 10:
                    break
                action = 'Begin'
                count = state[3]
                mdp.editBet(betPi[count])
                adjusted_state = ('', '', state[2], state[3])
                state = adjusted_state
            else:
                action = rl.getAction((state[0], state[1], count))

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
            print "Trial %d (totalReward = %s): %s" % (trial, totalReward, sequence)
        totalRewards.append(totalReward)
        totalStates.append(statesequence)


    return totalRewards,

def loadPolicy():
    policy = collections.defaultdict(str)
    count = [i for i in range(-10, 11)]
    for count in counts:
        current_pi = pickle.load(open('./policy/' + 'Count {} Policy.pkl'.format(count), 'rb'))
        policy.update(current_pi)
    return policy

def main():
    pi = loadPolicy()
    FixedRLAlgorithm(pi)
    trueMDP = ExactMDP.BlackjackMDP()




if __name__ == '__main__':
    main()
