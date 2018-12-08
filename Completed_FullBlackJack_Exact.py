import copy
import sys
import collections
from datetime import datetime
import cProfile
import re
import pickle
import math

class MDP:
    # Return the start state.
    def startState(self):
        raise NotImplementedError("Override me")

    # Return set of actions possible from |state|.
    def actions(self, state):
        raise NotImplementedError("Override me")

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    # Mapping to notation from class:
    #   state = s, action = a, newState = s', prob = T(s, a, s'), reward = Reward(s, a, s')
    # If IsEnd(state), return the empty list.
    def succAndProbReward(self, state, action):
        raise NotImplementedError("Override me")

    def discount(self):
        raise NotImplementedError("Override me")

    # Compute set of states reachable from startState.  Helper function for
    # MDPAlgorithms to know which states to compute values and policies for.
    # This function sets |self.states| to be the set of all states.
    def computeStates(self):
        self.states = set()
        queue = []
        self.states.add(self.startState())
        queue.append(self.startState())
        while len(queue) > 0:
            state = queue.pop()
            for action in self.actions(state):
                for newState, prob, reward in self.succAndProbReward(state, action):
                    if newState not in self.states:
                        self.states.add(newState)
                        queue.append(newState)
        # print "%d states" % len(self.states)


class BlackjackMDP(MDP):
    def __init__(self, cardValues=('2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'), multiplicity=1,
                 threshold=21, bet=1, blackjack=1.5, count=0):
        """
        cardValues: list of integers (face values for each card included in the deck)
        multiplicity: single integer representing the number of cards with each face value
        threshold: maximum number of points (i.e. sum of card values in hand) before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.bet = bet
        self.blackjack = blackjack
        self.count = count
        self.cardCount = (self.multiplicity,) * len(self.cardValues)

    def initialCardCount(self):
        return self.cardCount

    #####################################################################################################
    def startState(self):
        return ('', '', self.initialCardCount(), self.count)

    def sortCards(self, cardValue, deepAnalysis=True):
        if cardValue == '':
            return 0
        split = False
        double = False

        if type(cardValue) == int:
            return cardValue, False, 0, False

        elif cardValue[-1].isdigit():
            value = int(cardValue)
        else:
            for j in range(len(cardValue)):
                if not cardValue[j].isdigit():
                    value = int(cardValue[:j])
                    break
                else:
                    assert ValueError

        if deepAnalysis:
            if 'S' in cardValue:
                split = True
            if 'D' in cardValue:
                double = True

            acecounter = cardValue.count('A')
            return int(value), split, acecounter, double
        else:
            return int(value)

    #####################################################################################################
    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be placed into the succAndProbReward function below.
    def actions(self, state):
        cardValue, DealerCards, CardsRemaining, count = state

        if cardValue == '':
            return ['Begin']

        value, split, aces, double = self.sortCards(cardValue)

        if value >= 18 and not split and aces == 0:
            return ['Stay']

        elif double and split and (7 <= value <= 11 or 17 <= value <= 18):
            return ['Draw', 'Stay', 'Double']

        elif split:
            return ['Draw', 'Stay']  # Will implement split, double, later

        elif double and 6 <= value <= 11:
            return ['Draw', 'Double']

        elif double and 12 <= value <= 19 and aces > 0:
            return['Double', 'Stay', 'Draw']

        elif value <= 11:
            return ['Draw']

        elif type(value) == int:
            return ['Draw', 'Stay']

        else:
            raise NotImplemented('error')

    # Compute value of cards given the face value of each card from the state
    def cards_value(self, card_state):
        if type(card_state) == int:
            return card_state
        acecounter = card_state.count('A')
        if not card_state:
            return 0
        if card_state[-1].isdigit():
            value = int(card_state)
            return value
        else:
            value = int(''.join(n for n in card_state if n.isdigit()))

        while value > self.threshold and acecounter > 0:
            value -= 10
            acecounter -= 1
        return value

    def old_card_value(self, card_state):
        if card_state[-1].isdigit():
            return int(card_state), 0
        else:
            aces = card_state.count('A')
            value = ''.join(n for n in card_state if n.isdigit())
        return int(value), aces

    ###############################################################################################################
    def player_draw(self, card_state, CardsRemaining, prob_card):
        # Return hand of one or two cards, probabilities associated with each hand
        card_states = collections.defaultdict(float)
        card_value, aces = self.old_card_value(card_state)
        CardsRemaining_lst = list(CardsRemaining)

        for i, card in enumerate(self.cardValues):
            probability = prob_card[i]
            if probability != 0:
                CardsRemaining_lst[i] += -1
                new_card_state = self.createDrawState(card, card_value, acecounter=aces)
                card_states[(new_card_state, tuple(CardsRemaining_lst))] += probability
                CardsRemaining_lst[i] += 1
        return card_states

    def dealer_single_draw(self, dealerCards, CardsRemaining, prob_card):
        CardsRemaining_tup = tuple(CardsRemaining)
        cardValues = []
        remaining = []
        probability = []
        dealerCards = list(dealerCards)

        for i in range(len(self.cardValues)):
            if prob_card[i] != 0:
                CardsRemaining = list(CardsRemaining_tup)
                CardsRemaining[i] += -1
                tempvalues = copy.copy(dealerCards)
                tempvalues[i] = dealerCards[i] + 1
                cardValues.append(tuple(tempvalues))
                remaining.append(CardsRemaining)
                probability.append(prob_card[i])
        return tuple(cardValues), tuple(remaining), probability

    def createDrawState(self, cards, prev_val=0, specialplayer=False, acecounter=0):
        value = prev_val
        for card in cards:
            if card.isdigit():
                value += int(card)
            elif card == 'A':
                acecounter += 1
                value += 11
            else:
                value += 10
        while value > self.threshold and acecounter > 0:
            value -= 10
            acecounter -= 1
        state = str(value)
        state += acecounter * 'A'
        # Double and split options available if player has two cards
        if specialplayer:
            state += 'D'
            if cards[0] == cards[1]:
                state += 'S'
        return state

    def initial_draw(self, CardsRemaining):
        draws = []
        probabilities = []
        remaining = []

        for i in range(len(self.cardValues)):

            if CardsRemaining[i] > 0:
                counts = list(CardsRemaining)
                number = sum(CardsRemaining)
                prob_i = counts[i] / number
                counts[i] += -1
                number += -1

                for j in range(len(self.cardValues)):
                    if counts[j] > 0:
                        count1 = copy.copy(counts)
                        prob_j = count1[j] / number
                        count1[j] += -1
                        number += -1
                        for k in range(len(self.cardValues)):
                            if count1[k] > 0:
                                count2 = copy.copy(count1)
                                prob_k = count2[k] / number
                                number += -1
                                count2[k] += -1
                                for q in range(len(self.cardValues)):
                                    if count2[q] > 0:
                                        count3 = copy.copy(count2)
                                        prob_q = count3[q] / number
                                        count3[q] += -1
                                        remaining.append(count3)
                                        draws.append(self.cardValues[i] + self.cardValues[j] + self.cardValues[k] +
                                                     self.cardValues[q])
                                        probabilities.append(prob_i * prob_j * prob_k * prob_q)
                                number += 1
                        number += 1
                number += 1
        state_prob = collections.defaultdict(float)
        for i, draw in enumerate(draws):
            player = self.createDrawState(draw[0] + draw[2], specialplayer=True)
            dealer = self.createDrawState(draw[1] + draw[3], specialplayer=False)
            state_prob[(player, dealer, tuple(remaining[i]))] += probabilities[i]

        return state_prob

    def currentCount(self, cardsRemaining):
        return int(math.floor((sum(cardsRemaining[8:]) - sum(cardsRemaining[:5]))/sum(cardsRemaining)))

    def editBet(self, bet):
        self.bet = bet

    def succAndProbReward(self, state, action):
        result = []
        cardValue, dealerCards, CardsRemaining, _ = state
        player_value = self.cards_value(cardValue)

        # If number cards tuple is set to none, end state is reached and we return and empty result
        if dealerCards is None:
            return result

        # If the action is take, we enter here
        if action == 'Draw':
            prob_card = [float(i) / sum(CardsRemaining) for i in CardsRemaining]
            playerStates = self.player_draw(cardValue, CardsRemaining, prob_card)
            for i, key in enumerate(playerStates.keys()):
                # Put cardpairs in result with cardsRemaining, probabilities, no reward, or negative bet
                count = self.currentCount(key[1])
                if self.cards_value(key[0]) > 21:
                    result.append(((self.cards_value(key[0]), None, tuple(key[1]), count), playerStates[key], -self.bet))
                else:
                    result.append(((key[0], dealerCards, tuple(key[1]), count), playerStates[key], 0))
            return result

        # If action is stay, enter here
        elif action == 'Stay':
            # Draw all dealer cards, until over 17
            dealer_value = self.cards_value(dealerCards)
            queue = [(dealerCards, CardsRemaining, 1)]
            finalStates = collections.defaultdict(float)

            # If less than 17, dealer draws a card, we take action
            if dealer_value < 17:
                while len(queue) != 0:
                    currentState = queue.pop(0)
                    probability = [float(i) / sum(currentState[1]) for i in currentState[1]]
                    newDealerStates = self.player_draw(currentState[0], currentState[1], probability)
                    for state, prob in newDealerStates.items():
                        new_value = self.cards_value(state[0])
                        if new_value < 17:
                            queue.append((state[0], state[1], currentState[2] * prob))
                        else:
                            finalStates[(new_value, state[1], self.currentCount(state[1]))] += currentState[2] * prob
            else:
                finalStates[(self.cards_value(queue[0][0]), self.currentCount(queue[0][1]))] = 1

            for values, prob in finalStates.items():
                dealervalue, cards, count = values
                count = self.currentCount()
                if 17 <= dealervalue <= 21:
                    if dealervalue > player_value:
                        result.append(((player_value, None, cards, count), prob, -self.bet))
                    elif dealervalue < player_value:
                        result.append(((player_value, None, cards, count), prob, self.bet))
                    else:
                        result.append(((player_value, None, cards, count), prob, 0))

                elif dealervalue > 21:
                    result.append(((player_value, None, cards, count), prob, self.bet))
            return result

        elif action == 'Begin':
            state_prob = self.initial_draw(CardsRemaining)

            for key in state_prob:
                count = self.currentCount(key[2])
                playervalue = self.cards_value(key[0])
                dealervalue = self.cards_value(key[1])

                if dealervalue == 21 and playervalue == 21:
                    result.append(((key[0], key[1], key[2], count), state_prob[key]), 0)
                elif playervalue == 21 and dealervalue != 21:
                    result.append(((21, dealervalue, key[2], count), state_prob[key], self.blackjack * self.bet))

                elif playervalue != 21 and dealervalue == 21:
                    result.append(((playervalue, dealervalue, key[2], count), state_prob[key], -self.bet))

                elif playervalue < 21 and dealervalue < 21:
                    result.append(((key[0], key[1], key[2], count), state_prob[key], 0))
            return result

        elif action == 'Double':
            # Calculate player probability values
            prob_card = [float(i) / sum(CardsRemaining) for i in CardsRemaining]
            playerStates = self.player_draw(cardValue, CardsRemaining, prob_card)

            for playerstate in list(playerStates.keys()):
                count = self.currentCount(playerstate[1])
                val = self.cards_value(playerstate[0])
                if val > 21:
                    result.append(((playerstate[0], None, playerstate[1], count), playerStates[playerstate], -2*self.bet))
                    del playerStates[playerstate]

            # Calculate dealer combinations and probability values
            queue = []
            dealer_value = self.cards_value(dealerCards)
            finalStates = collections.defaultdict(float)


            # If less than 17, dealer draws a card, we take action
            if dealer_value < 17:
                for states, prob in playerStates.items():
                    finalStates = collections.defaultdict(float)
                    queue.append((dealerCards, states[1], prob))
                    while len(queue) > 0:
                        currentState = queue.pop(0)
                        probability = [float(i) / sum(currentState[1]) for i in currentState[1]]
                        newDealerStates = self.player_draw(currentState[0], currentState[1], probability)

                        for state, probdealer in newDealerStates.items():
                            new_value = self.cards_value(state[0])
                            if new_value < 17:
                                queue.append((state[0], state[1], currentState[2] * probdealer))
                            else:
                                finalStates[new_value, state[1], self.currentCount(state[1])] += currentState[2] * probdealer

                    playervalue = self.cards_value(states[0])

                    for values, prob_dealer in finalStates.items():
                        dealervalue, cardsleft, count = values
                        if dealervalue > 21:
                            result.append(((playervalue, None, cardsleft, count), prob_dealer, 2 * self.bet))
                        elif dealervalue < playervalue:
                            result.append(((playervalue, None, cardsleft, count), prob_dealer, 2 * self.bet))
                        elif dealervalue > playervalue:
                            result.append(((playervalue, None, cardsleft, count), prob_dealer, -2 * self.bet))
                        elif playervalue == dealervalue:
                            result.append(((playervalue, None, cardsleft, count), prob_dealer, 0))

            else:
                for playerstate, prob_player in playerStates.items():
                    playervalue = self.cards_value(playerstate[0])
                    count = self.currentCount(playerstate[2])
                    if dealer_value > 21:
                        raise NotImplemented('Error')
                    elif dealer_value < playervalue:
                        result.append(((playervalue, None, playerstate[2], count), prob_player, 2 * self.bet))
                    elif dealer_value > playervalue:
                        result.append(((playervalue, None, playerstate[2], count), prob_player, -2 * self.bet))
                    elif playervalue == dealer_value:
                        result.append(((playervalue, None, playerstate[2], count), prob_player, 0))
            return result

        else:
            raise ValueError("Shouldn't be calling Dealer Draw Here")

    def discount(self):
        return 1


class MDPAlgorithm:
    # Set:
    # - self.pi: optimal policy (mapping from state to action)
    # - self.V: values (mapping from state to best values)
    def solve(self, mdp): raise NotImplementedError("Override me")


class ValueIteration(MDPAlgorithm):
    '''
    Solve the MDP using value iteration.  Your solve() method must set
    - self.V to the dictionary mapping states to optimal values
    - self.pi to the dictionary mapping states to an optimal action
    Note: epsilon is the error tolerance: you should stop value iteration when
    all of the values change by less than epsilon.
    The ValueIteration class is a subclass of util.MDPAlgorithm (see util.py).
    '''
    def solve(self, mdp, epsilon=0.001):
        print('Computing States:')
        startTime = datetime.now()
        mdp.computeStates()
        print('Took', datetime.now() - startTime, 'to compute', len(mdp.states), 'states')
        def computeQ(mdp, V, state, action):
            # Return Q(state, action) based on V(state).
            return sum(prob * (reward + mdp.discount() * V[newState]) \
                            for newState, prob, reward in mdp.succAndProbReward(state, action))

        def computeOptimalPolicy(mdp, V):
            # Return the optimal policy given the values V.
            pi = {}
            for state in mdp.states:
                pi[state] = max((computeQ(mdp, V, state, action), action) for action in mdp.actions(state))[1]
            return pi

        V = collections.defaultdict(float)  # state -> value of state
        numIters = 0
        print('Value Iteration Process')
        print('Iteration {}'.format(numIters+1))
        while True:
            newV = {}
            for state in mdp.states:
                # This evaluates to zero for end states, which have no available actions (by definition)
                newV[state] = max(computeQ(mdp, V, state, action) for action in mdp.actions(state))
            numIters += 1
            print('Error = ', max(abs(V[state] - newV[state]) for state in mdp.states))
            print('Iteration {}'.format(numIters + 1))
            if max(abs(V[state] - newV[state]) for state in mdp.states) < epsilon:
                V = newV
                break
            V = newV

        # Compute the optimal policy now
        pi = computeOptimalPolicy(mdp, V)
        print("ValueIteration: %d iterations" % numIters)
        self.pi = pi
        self.V = V

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
    # count = mdp.initialCardCount()
    # checkdouble = True
    # checkbegin = True
    # checkdraw = True
    # checkstay = True
    #
    # if checkdouble:
    #     result = mdp.succAndProbReward(('12D', '16', count), 'Double')
    #     print('Check Double', result, sum(x[1] for x in result))
    #
    # elif checkbegin:
    #     result = mdp.succAndProbReward(('', '', count), 'Begin')
    #     print('Check Begin', result)
    # elif checkdraw:
    #     result = mdp.succAndProbReward(('4', '4', count), 'Draw')
    #     print('Check Draw', result)
    # elif checkstay:
    #     result = mdp.succAndProbReward(('17', '12', count), 'Stay')
    #     print('Check Stay', result)
    mdp = BlackjackMDP()

    for i in range(4):
        mdp = BlackjackMDP(multiplicity=i+1)
        startState = mdp.startState()
        alg = ValueIteration()
        print('Algorithm Value iteration with multiplicity {}'.format(i+1))
        alg.solve(mdp, .001)
        save_obj('Multiplicity {} Policy'.format(i), alg.pi)
        print('Saved policty for multiplicity {}'.format(i+1))
        save_obj('Multiplicity {} V'.format(i), alg.V)
        print('Saved Expected Value Value for multiplicity {}'.format(i+1))

        print(alg.V[startState])

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






if __name__ == '__main__':
    main()