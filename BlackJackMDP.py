import copy
import sys
import collections
from datetime import datetime
import re
import pickle
import itertools
import datetime

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
    """
    Creates BlackJackMDP from parent class MDP, inherits 'computeStates' function which computes all
    relevant states and returns them as a set for the MDP through a queue search.
    """
    def __init__(self, cardValues=('2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'),
                 threshold=21, bet=1, count=0, blackjack=1.5, numCards=52, midcards = 12):
        """
        cardValues:     list of integers (face values for each card included in the deck)
        multiplicity:   single integer representing the number of cards with each face value
        threshold:      maximum number of points (i.e. sum of card values in hand) before going bust
        peekCost:       how much it costs to peek at the next card
        blackjack:      multiplier if blackjack is achieved with 2 cards for player
        numCards:       Total number of cards in the deck
        midcards:       Number of 7-9 value cards in the deck. These are not taken into account in general definition of
                        "True Count", thus provide variability in expected value despite same count value
        cardLevels:     List of 3 tuples. First tuple holds cards associated w/ -1 count, second tuple hold cards
                        assoc. w/ 0 count, third holds values w/ +1 count
        probability:    Calculates probability of drawing certain card value from different levels in "cardLevels"
        pl/pm/ph        Ind probability of lower card, mid card, and high card
        probabilitycard Prob of certain card str call happening
        """
        self.cardValues = cardValues
        self.threshold = threshold
        self.bet = bet
        self.cardcount = count
        self.blackjack = blackjack
        self.numCards = numCards
        self.initialcount = count
        self.cardLevels = [(self.cardValues[:5]), (self.cardValues[5:8]), (self.cardValues[8:])]
        self.probability = [(52-midcards-count)/2/self.numCards/5, midcards/self.numCards/3,
                            (count+(52-midcards-count)/2)/self.numCards/5]
        self.pl = (52-midcards-count)/2/self.numCards/5
        self.pm = midcards/self.numCards/3
        self.ph = (count + (52-midcards-count)/2)/self.numCards/5
        self.probabilitycard = {'2': self.pl, '3': self.pl, '4': self.pl, '5': self.pl, '6': self.pl, '7': self.pm,
                                '8': self.pm, '9': self.pm, 'T': self.ph, 'J': self.ph, 'Q': self.ph,
                                'K': self.ph, 'A': self.ph}

    def startState(self):
        """Returns the start state of the MDP problem"""
        return ('', '', self.cardcount)

    def actions(self, state):
        """
        Given a state, return all viable actions for the MDP. Using logic, took out some of the actions for double to
        shrink state space and increase efficiency. These actions that were taking out had very little chance in
        doing the policy.
        """

        cardValue, DealerCards, count = state
        if cardValue == '':
            return ['Begin']
        if count is None:
            return [None]
        value, split, aces, double = self.sortCards(cardValue)
        if cardValue == 21:
            return ['Stay']
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

    def sortCards(self, cardValue, deepAnalysis=True):
        """
        Summary: Given a card state, this function will output the value of the cards and num aces present
                    and whether it is valid to split or double your current hand. Used specifically in the actions
                    function to gather information on which action can be taken in the MDP.

        :param cardValue: Input card state.
        :param deepAnalysis: Given an input state, will return if state can be doubled, split, and teh number of aces.
                                If false, will not return these
        :return: Will return the value, split boolean, acecounter value, and double boolean.
        """

        if cardValue == '':
            return 0
        split = False
        double = False
        x = 1

        if type(cardValue) == int:
            return cardValue, False, 0, False

        elif '*' not in cardValue:
            value = int(cardValue)

        else:
            x = cardValue.find('*')
            value = int(cardValue[:x])

        # If deepAnalysis is turned on, will also return the split and double boolean values along with acecount.
        if deepAnalysis:
            if 'S' in cardValue[x:]:
                split = True
            if 'D' in cardValue[x:]:
                double = True

            acecounter = cardValue.count('A')
            return int(value), split, acecounter, double
        else:
            return int(value)

    def cards_value(self, card_state, dealerintial=False):
        """
        Takes a card state inputted, and ouputs integer value of the card state inputted. Can toggle in inputs if this
        is a dealer hand or a player hand. Dealer hand is more simplified and can be done more efficiently.

        :param card_state:      Card state value plugged in, directly from the first entry of the state tuple
        :param dealerintial:    If this option is turned on, returns card value specifically for a dealer state input.
                                Otherwise, player input is assumed. Dealer state more simplified than player state.
        :return:                Returns the integer value of the card state inputted. Takes into account aces.
        """
        # If dealer hand toggle is on, run following simplified cod for dealer analysis
        if dealerintial:
            if card_state == 11:
                return str(card_state) + '*A'
            return card_state
        if type(card_state) == int:
            return card_state
        acecounter = card_state.count('A')
        if not card_state:
            return 0
        symindex = card_state.find('*')
        if symindex >= 0:
            value = int(card_state[:symindex])
            return value
        else:
            value = int(card_state)

        while value > self.threshold and acecounter > 0:
            value -= 10
            acecounter -= 1
        return value

    def old_card_value(self, card_state):
        """
        Given a card_state, returns a card value and the number of aces
        """
        if type(card_state) == int:
            return card_state, 0

        acecounter = card_state.count('A')
        if not card_state:
            return 0, 0
        symindex = card_state.find('*')

        if symindex >= 0:
            value = int(card_state[:symindex])

        else:
            value = int(card_state)

        while value > self.threshold and acecounter > 0:
            value -= 10
            acecounter -= 1

        return value, acecounter

    ###############################################################################################################
    def player_draw(self, card_state):
        """
        Given a initial state, return dict of all possible new states and the probability that each state will happen.

        :param card_state: Current card state of the player or dealer
        :return: Returns dict of all possible new card states with a value of the probability that a certain state will
                happen. All probabilities in the dict will add up to one, full representation or probability outcomes.
        """
        # Return hand of one or two cards, probabilities associated with each hand
        card_states = collections.defaultdict(float)
        card_value, aces = self.old_card_value(card_state)

        for i, cards in enumerate(self.cardLevels):
            prob = self.probability[i]
            for card in cards:
                if prob > 0:
                    new_card_state = self.createDrawState(card, card_value, acecounter=aces)
                    card_states[new_card_state] += prob
        return card_states

    def createDrawState(self, cards, prev_val=0, specialplayer=False, acecounter=0, dealerinitial=False):
        """
        Create player or dealer draw state based on the card value and appropriate indicators.

        Indicators:
        A = player holds an ace in their hand with value 11. If player hold ace with value one, no indicator present
        D = Player holds 2 cards and can double hand
        S = Player holds 2 cards that are the same, can split hand


        :param cards: Card keys of new cards drawn. (i.e. '2', 'k', etc.)
        :param prev_val: Integer of previous value of hand. (i.e. 18)
        :param specialplayer: If special player toggle is on, add in double indicator and check split indicator
        :param acecounter: Number of aces found in the previous hand
        :param dealerinitial: If this toggle turned on, return an int of the card value. Only one card should be inputted
                                since only only one card in the dealers hand is seen.
        :return: Return draw state with appropriate indicators. For player, draw state is string with indicators. For dealer
                    draw state is integer for initial card, and string otherwise.
        """
        if dealerinitial:
            if cards.isdigit():
                return int(cards)
            elif cards == 'A':
                return 11
            else:
                return 10

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
        state += '*' + acecounter * 'A'

        # Double and split options available if player has two cards
        if specialplayer:
            state += 'D'
            if cards[0] == cards[1]:
                state += 'S'
        return state

    def initial_draw(self):
        """
        Uses itertool package to find all possible combination of cards that can be drawn in initial draw of 3 cards
        (2 to the player 1 to the dealer). Subsequent probabilities are found for each hand.

        :return: Returns dict of possible player and dealer states with value equal to the probability that a certain state
                    will be found.
        """
        combos = list(itertools.product(self.cardValues, repeat=3))
        states = collections.defaultdict(float)
        for cards in combos:
            prob = 1
            for card in cards:
                prob *= self.probabilitycard[card]
            player = self.createDrawState(cards[0] + cards[2], specialplayer=True)
            dealer = self.createDrawState(cards[1], specialplayer=False, dealerinitial=True)
            states[(player, dealer)] += prob
        return states

    def succAndProbReward(self, state, action):
        result = []
        cardValue, dealerCards, count = state
        player_value = self.cards_value(cardValue)

        # If number cards tuple is set to none, end state is reached and we return and empty result
        if count is None:
            return result

        # If the action is take, we enter here
        if action == 'Draw':
            playerStates = self.player_draw(cardValue)
            for key in playerStates.keys():
                # Put cardpairs in result with cardsRemaining, probabilities, no reward, or negative bet
                if self.cards_value(key) > 21:
                    result.append(((self.cards_value(key), None, None), playerStates[key], -self.bet))
                else:
                    result.append(((key, dealerCards, self.cardcount), playerStates[key], 0))
            return result

        # If action is stay, enter here
        elif action == 'Stay':

            if cardValue == 21:
                dealerStates = self.player_draw(dealerCards)
                for key in dealerStates.keys():
                    # Put cardpairs in result with cardsRemaining, probabilities, no reward, or negative bet
                    dealer_value = self.cards_value(key)
                    if cardValue == 21:
                        if dealer_value < 21:
                            result.append(((cardValue, None, None), dealerStates[key], self.blackjack*self.bet))
                        elif dealer_value == 21:
                            result.append(((21, 21, None), dealerStates[key], 0))
                return result

            else:
                queue = []
                finalStates = collections.defaultdict(float)
                if dealerCards == 11:
                    dealerCards = str(dealerCards) + '*A'

                dealerStates = self.player_draw(dealerCards)
                for key in dealerStates:
                    # Put cardpairs in result with cardsRemaining, probabilities, no reward, or negative bet
                    dealer_value = self.cards_value(key)
                    if dealer_value == 21:
                        result.append(((cardValue, 21, None), dealerStates[key], -self.bet))
                    elif dealer_value >= 17:
                        finalStates[dealer_value] += dealerStates[key]
                    else:
                        queue.append((key, dealerStates[key]))

            # Draw all dealer cards, until over 17

            # If less than 17, dealer draws a card, we take action
            while len(queue) != 0:
                currentState = queue.pop(0)
                newDealerStates = self.player_draw(currentState[0])
                for state, prob in newDealerStates.items():
                    new_value = self.cards_value(state)
                    if new_value < 17:
                        queue.append((state, currentState[1] * prob))
                    else:
                        finalStates[new_value] += currentState[1] * prob

            for dealervalue, prob in finalStates.items():
                if 17 <= dealervalue <= 21:
                    if dealervalue > player_value:
                        result.append(((player_value, dealervalue, None), prob, -self.bet))
                    elif dealervalue < player_value:
                        result.append(((player_value, dealervalue, None), prob, self.bet))
                    else:
                        result.append(((player_value, dealervalue, None), prob, 0))

                elif dealervalue > 21:
                    result.append(((player_value, dealervalue, None), prob, self.bet))

                else:
                    raise ValueError('Shouldnt have dealer less than this value')
            return result

        elif action == 'Begin':
            state_prob = self.initial_draw()

            for key in state_prob:
                playervalue = self.cards_value(key[0])
                dealervalue = self.cards_value(key[1])

                if playervalue == 21:
                    result.append(((21, key[1], self.cardcount), state_prob[key], 0))

                elif playervalue < 21:
                    result.append(((key[0], key[1], self.cardcount), state_prob[key], 0))

                elif playervalue > 21 or dealervalue > 21:
                    raise ValueError("Yo, u screwed up, can't get over 21 with 2 cards")

                else:
                    raise ValueError('Why you in here line 426')
            return result

        elif action == 'Double':
            # Calculate player probability values
            playerStates = self.player_draw(cardValue)
            queue = []
            for playerstate in list(playerStates.keys()):
                val = self.cards_value(playerstate)
                if val > 21:
                    result.append(((playerstate, None, None), playerStates[playerstate], -2*self.bet))
                    del playerStates[playerstate]
            # Calculate dealer combinations and probability values
            finalStates = collections.defaultdict(float)

            if dealerCards == 11:
                dealerCards = str(dealerCards) + '*A'
            dealerStates = self.player_draw(dealerCards)
            sum = 0

            for key in dealerStates:
                # Put cardpairs in result with cardsRemaining, probabilities, no reward, or negative bet
                dealer_value = self.cards_value(key)
                if dealer_value == 21:
                    result.append(((cardValue, 21, None), dealerStates[key], -self.bet))
                    sum += dealerStates[key]
                elif dealer_value >= 17:
                    finalStates[dealer_value] += dealerStates[key]
                else:
                    queue.append((key, dealerStates[key]))

            while len(queue) > 0:
                currentState = queue.pop(0)
                newDealerStates = self.player_draw(currentState[0])
                for state, probdealer in newDealerStates.items():
                    new_value = self.cards_value(state)
                    if new_value < 17:
                        queue.append((state, currentState[1] * probdealer))
                    else:
                        finalStates[new_value] += currentState[1] * probdealer

            # If less than 17, dealer draws a card, we take action
            for states, prob_player in playerStates.items():
                playervalue = self.cards_value(states)
                for dealervalue, prob_dealer in finalStates.items():
                    if dealervalue > 21:
                        result.append(((playervalue, dealervalue, None), prob_dealer*prob_player, 2 * self.bet))
                    elif dealervalue < playervalue:
                        result.append(((playervalue, dealervalue, None), prob_dealer*prob_player, 2 * self.bet))
                    elif dealervalue > playervalue:
                        result.append(((playervalue, dealervalue, None), prob_dealer*prob_player, -2 * self.bet))
                    elif playervalue == dealervalue:
                        result.append(((playervalue, dealervalue, None), prob_dealer*prob_player, 0))
            return result

        elif action == 'DealerSingle':
            dealerStates = self.player_draw(dealerCards)
            for key in dealerStates.keys():
                # Put cardpairs in result with cardsRemaining, probabilities, no reward, or negative bet
                dealer_value = self.cards_value(key[0])
                if cardValue == 21:
                    if dealer_value < 21:
                        result.append(((cardValue, None, None), dealerStates[key], self.blackjack*self.bet))
                    elif dealer_value == 21:
                        result.append(((21, 21, None), dealerStates[key], 0))
            return result

    def discount(self):
        return 1
