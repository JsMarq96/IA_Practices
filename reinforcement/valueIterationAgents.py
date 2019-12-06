# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):

        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
       # for state in mdp.getStates():
        #  for action in mdp.getPossibleActions(state):
         #   print(mdp.getTransitionStatesAndProbs(state, action), state)

        result = []

        stack = [self.mdp.getStartState()]

        for state in stack:
            result = []
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    result.append(self.computeQValueFromValues(state,action))

                    for state, _ self.mdp.getTransitionStatesAndProbs(state, action):
                self.values[state] = max(result)

        for i in range(self.iterations):
            for j, state in enumerate(self.mdp.getStates()):
                if j > i:
                    break
                result = []
                if not self.mdp.isTerminal(state):
                    for action in self.mdp.getPossibleActions(state):
                        result.append(self.computeQValueFromValues(state,action))
                    self.values[state] = max(result)
        print(self.iterations)
        print(self.values)
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        result = []
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, next_state) + (self.discount * self.values[next_state])
            #reward += (self.discount * self.values[next_state])
            result.append(reward * prob)

        return sum(result)
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """

        result = []
        if self.mdp.isTerminal(state):
           return 'exit'

        for action in self.mdp.getPossibleActions(state):
            result.append((self.computeQValueFromValues(state,action), action))


        return max(result)[1]
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)