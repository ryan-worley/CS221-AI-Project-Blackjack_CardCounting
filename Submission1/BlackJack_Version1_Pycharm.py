import copy
import sys
import collections


class MDP:
    # Return the start state.
    def startState(self): raise NotImplementedError("Override me")

    # Return set of actions possible from |state|.
    def actions(self, state): raise NotImplementedError("Override me")

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    # Mapping to notation from class:
    #   state = s, action = a, newState = s', prob = T(s, a, s'), reward = Reward(s, a, s')
    # If IsEnd(state), return the empty list.
    def succAndProbReward(self, state, action): raise NotImplementedError("Override me")

    def discount(self): raise NotImplementedError("Override me")

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
                    print(newState)
                    if newState not in self.states:
                        self.states.add(newState)
                        queue.append(newState)
        # print "%d states" % len(self.states)
        print(self.states)


class BlackjackMDP(MDP):
    def __init__(self, cardValues=('2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'), multiplicity=4,
                 threshold=21, bet=1, blackjack=1.5):
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

    # Return the start state.
    # Look closely at this function to see an example of state representation for our Blackjack game.
    # Each state is a tuple with 3 elements:
    #   -- The first element of the tuple is the sum of the cards in the player's hand.
    #   -- If the player's last action was to peek, the second element is the index
    #      (not the face value) of the next card that will be drawn; otherwise, the
    #      second element is None.
    #   -- The third element is a tuple giving counts for each of the cards remaining
    #      in the deck, or None if the deck is empty or the game is over (e.g. when
    #      the user quits or goes bust).
    def initialCardCount(self):
        return (self.multiplicity,) * len(self.cardValues)

    #####################################################################################################
    def startState(self):
        return ('', '', self.initialCardCount())

    #####################################################################################################
    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be placed into the succAndProbReward function below.
    def actions(self, state):
        cardValue, DealerCards, CardsRemaining = state

        if cardValue == '':
            return ['Draw']

        elif DealerCards == '':
            return ['DealerDraw']

        elif len(cardValue) == 1 and len(DealerCards) == 1:
            return ['Draw']

        elif len(cardValue) == 2 and len(DealerCards) == 1:
            return ['DealerDraw']

        elif cardValue[0] == cardValue[1] and len(cardValue) == 2:
            return ['Draw', 'Stay']  # Will implement split, double, later

        elif len(cardValue) == 2:
            return ['Draw', 'Stay']  # Double

        elif len(DealerCards) >= 3:
            return ['Stay']

        return ['Draw', 'Stay']

    # Compute value of cards given the face value of each card from the state
    def cards_value(self, cards):
        acecounter = 0
        cardvalue = 0
        for card in cards:
            if card in ['T', 'J', 'Q', 'K']:
                cardvalue += 10
            elif card == 'A':
                acecounter += 1
                cardvalue += 11
            else:
                cardvalue += int(card)

        while cardvalue > self.threshold and acecounter > 0:
            cardvalue -= 10
            acecounter -= 1
        return cardvalue

    ###############################################################################################################
    def player_draw(self, cardValue, CardsRemaining, prob_card):
        # Return hand of one or two cards, probabilities associated with each hand
        CardsRemaining_tup = tuple(CardsRemaining)
        # If no cards drawn yet
        cardvalues = []
        probability = []
        remaining = []
        CardsRemaining = list(CardsRemaining)

        #         # Draw 2 cards
        #         if cardValue is None:
        #             for i in range(len(self.cardValues)):
        #                 CardsRemaining = list(CardsRemaining_tup)

        #                 if prob_card[i] != 0:
        #                     CardsRemaining[i] += -1
        #                     prob_1 = prob_card[i]
        #                     card = self.cardValues[i]

        #                 # If there is not probability of drawing certain card, do not enter
        #                     for j in range(len(self.cardValues)):
        #                         if prob_card[j] != 0:
        #                             newRemaining = copy.copy(CardsRemaining)
        #                             newRemaining[j] += -1
        #                             prob_card_new = [float(k) / sum(newRemaining) for k in newRemaining]
        #                             cardValue = card + self.cardValues[j]
        #                             cardvalues.append(cardValue)
        #                             prob_2 = prob_card_new[j]
        #                             probability.append(prob_1*prob_2)
        #                             remaining.append(newRemaining)
        #             return cardvalues, tuple(remaining), probability

        for i in range(len(self.cardValues)):
            if prob_card[i] != 0:
                CardsRemaining = list(CardsRemaining_tup)
                CardsRemaining[i] += -1
                cardvalues.append(cardValue + self.cardValues[i])
                remaining.append(CardsRemaining)
                probability.append(prob_card[i])
        return cardvalues, tuple(remaining), probability

    ##############################################################################################################
    def dealer_continuous_draw(self, dealerShow, CardsRemaining, prob_card, old_prob):
        # Wowie
        cardvalues = []
        new_probability = []
        remaining = []
        CardsRemaining = list(CardsRemaining)

        for j in range(len(dealerShow)):
            for i in range(len(self.cardValues)):
                if prob_card[i] != 0:
                    CardsRemaining[i] += -1
                    cardvalues.append(dealerShow + self.cardValues[j])
                    remaining.append(CardsRemaining)
                    new_probability.append(prob_card[i] * old_prob[j])

        return cardvalues, tuple(CardsRemaining), new_probability

    def dealer_single_draw(self, dealerCards, CardsRemaining, prob_card):
        CardsRemaining_tup = tuple(CardsRemaining)
        cardValues = []
        remaining = []
        probability = []

        for i in range(len(self.cardValues)):
            if prob_card[i] != 0:
                CardsRemaining = list(CardsRemaining_tup)
                CardsRemaining[i] += -1
                cardValues.append(dealerCards + self.cardValues[i])
                remaining.append(CardsRemaining)
                probability.append(prob_card[i])
        return cardValues, tuple(remaining), probability

    # State (cardValues, DealerShownCards, CardsRemaining)
    # Other ()

    # Cardvalues: List of cards in players hands
    # Dealer Shown Cards: Card that is shown by the dealer
    # Cards Remaining

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # A few reminders:
    # * Indicate a terminal state (after quitting, busting, or running out of cards)
    #   by setting the deck to None.
    # * If |state| is an end state, you should return an empty list [].
    # * When the probability is 0 for a transition to a particular new state,
    #   don't include that state in the list returned by succAndProbReward.

    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 53 lines of code, but don't worry if you deviate from this)

        result = []
        cardValue, dealerCards, CardsRemaining = state

        # If numcards tuple is set to none, end state is reached and we return and empty result
        if CardsRemaining is None:
            return result

        # Do front end calculation of probability of each card being drawn
        prob_card = [float(i) / sum(CardsRemaining) for i in CardsRemaining]

        ###################################################################################################################
        # If the action is take, we enter here
        if action == 'Draw':
            # If the peek_index hasn't been used, we find probability of next cards being drawn

            cardpairs, CardsRemaining, probabilities = self.player_draw(cardValue, CardsRemaining, prob_card)

            if len(cardpairs[1]) <= 2:
                for i in range(len(cardpairs)):
                    # Put cardpairs in result with cardsRemaining, probabilities, no reward, or negative bet
                    if self.cards_value(cardpairs[i]) > 21:
                        raise ValueError("Yo, u screwed up, can't get over 21 with 2 cards")
                    result.append(((cardpairs[i], dealerCards, CardsRemaining[i]), probabilities[i], 0))
                return result

            else:
                for i in range(len(cardpairs)):
                    # Put cardpairs in result with cardsRemaining, probabilities, no reward, or negative bet
                    if self.cards_value(cardpairs[i]) > 21:
                        result.append(((cardpairs[i], dealerCards, None), probabilities[i], -self.bet))
                    else:
                        result.append(((cardpairs[i], dealerCards, CardsRemaining[i]), probabilities[i], 0))
                return result

                #####################################################################################################################
        # We peek, measure probability of which peek card we see, employ peek cost
        elif action == 'Stay':
            probability = [1]

            # Draw all dealer cards, until over 17
            dealervalue = self.cards_value(dealerCards)
            playervalue = self.cards_value(cardValue)

            # If less than 17, dealer draws a card, we take action
            if int(dealervalue) < 17:
                newdealerCards, CardsRemaining, probability = self.dealer_single_draw(dealerCards, CardsRemaining,
                                                                                      prob_card)

                for i in range(len(newdealerCards)):
                    dealervalue = self.cards_value(newdealerCards[i])

                    if 17 <= dealervalue <= 21:
                        if dealervalue > playervalue:
                            result.append(((cardValue, newdealerCards[i], None), probability[i], -self.bet))
                        elif dealervalue < playervalue:
                            result.append(((cardValue, newdealerCards[i], None), probability[i], self.bet))
                        else:
                            result.append(((cardValue, newdealerCards[i], None), probability[i], 0))

                    elif dealervalue > 21:
                        result.append(((cardValue, newdealerCards[i], None), probability[i], self.bet))

                    elif dealervalue < 17:
                        result.append(((cardValue, newdealerCards[i], CardsRemaining[i]), probability[i], 0))

                return result

            # If dealer over 17, compare values to player
            else:
                if dealervalue >= 21: Raise
                ValueError('Dealer Shouldnt be over 21 here')

                if dealervalue > playervalue:
                    result.append(((cardValue, dealerCards, None), 1, -self.bet))
                elif dealervalue < playervalue:
                    result.append(((cardValue, dealerCards, None), 1, self.bet))
                else:
                    result.append(((cardValue, dealerCards, None), 1, 0))
                return result

            #                 for i in range(len(DealerShownCards)):
        #                     dealervalues = self.cards_value(DealerShownCards[i])
        #                     prob_card = [float(i) / sum(CardsRemaining) for i in CardsRemaining]

        #             for i in range(len(dealervalues)):
        #                 if dealervalues[i] > self.threshold:
        #                     # player wins, reward bet
        #                     result.append(((cardValue, dealerShownCard, None), probability[i], self.bet))

        #                 elif dealervalues[i] == self.cards_value(cardValue):
        #                     # Tie, reward bet
        #                     result.append(((cardValue, dealerShownCard, None), probability[i], 0))

        #                 elif dealervalues[i] < self.cards_value(cardValue):
        #                     # Player Wins Bet, reward +bet
        #                     result.append(((cardValue, dealerShownCard, None), probability[i], self.bet))

        #                 else:
        #                     # Dealer wins
        #                     result.append(((cardValue, dealerShownCard, None), probability[i], -self.bet))

        #####################################################################################################################

        elif action == 'DealerDraw':
            dealercards, CardsRemaining, probabilities = self.dealer_single_draw(dealerCards, CardsRemaining, prob_card)

            if len(dealercards[1]) == 1:
                for i in range(len(dealercards)):
                    # Put cardpairs in result with cardsRemaining, probabilities, no reward, or negative bet
                    result.append(((cardValue, dealercards[i], None), probabilities[i], 0))
                return result

            elif len(dealercards[1]) == 2:
                for i in range(len(dealercards)):
                    # Put cardpairs in result with cardsRemaining, probabilities, no reward, or negative bet
                    playervalue = self.cards_value(cardValue)
                    dealervalue = self.cards_value(dealercards[i])

                    if playervalue > 21:
                        raise ValueError("Yo, u screwed up, can't get over 21 with 2 cards")

                    elif dealervalue == 21 and playervalue == 21:
                        result.append(((cardValue, dealercards[i], None), probabilities[i], 0))

                    elif playervalue == 21 and dealervalue != 21:
                        result.append(((cardValue, dealercards[i], None), probabilities[i], self.blackjack * self.bet))

                    elif playervalue != 21 and dealervalue == 21:
                        result.append(((cardValue, dealercards[i], None), probabilities[i], -self.bet))

                    elif playervalue < 21 and dealervalue < 21:
                        result.append(((cardValue, dealercards[i], CardsRemaining), probabilities[i], 0))

                    else:
                        raise ValueError("Shouldn't be anything over 21 here")
                else:
                    ValueError("Shouldn't be calling Dealer Draw Here")
                return result

        else:
            raise ValueError("Should have all actions covered")


#####################################################################################################################

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
        mdp.computeStates()
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
        while True:
            newV = {}
            for state in mdp.states:
                # This evaluates to zero for end states, which have no available actions (by definition)
                newV[state] = max(computeQ(mdp, V, state, action) for action in mdp.actions(state))
            numIters += 1
            if max(abs(V[state] - newV[state]) for state in mdp.states) < epsilon:
                V = newV
                break
            V = newV

        # Compute the optimal policy now
        pi = computeOptimalPolicy(mdp, V)
        print("ValueIteration: %d iterations" % numIters)
        self.pi = pi
        self.V = V

def main():
    mdp = BlackjackMDP()
    alg = ValueIteration()
    startState = mdp.startState()
    alg.solve(mdp, .001)
    print(alg.V(startState))

if __name__ == '__main__':
    main()