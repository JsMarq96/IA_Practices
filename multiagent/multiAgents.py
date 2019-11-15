# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        def stateValue(state):
          '''
          Function to calculate a puntuation of the inputed state
          '''
          newPos = state.getPacmanPosition()
          newFood = state.getFood()
          newGhostStates = state.getGhostStates()
          newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
          ghost_positions = [g_state.getPosition()  for g_state in state.getGhostStates()]


          food_dist  = [manhattanDistance(newPos, x) for x in newFood.asList()]

          ghost_dist = [manhattanDistance(newPos, x) for x in ghost_positions]

          score = -len(newFood.asList())

          # Emergency Ghost
          # Penalizes heavily if there is a ghost near
          very_near_ghost = sum([ -99  for x in ghost_dist if x < 2])

          nearest_ghost = min(ghost_dist)

          # more penalization for the near ghost, in a range from Pacman
          if nearest_ghost >= 6:
            nearest_ghost = 999
          else:
            nearest_ghost = 0

          # Get the distance from the enarest food
          nearest_coin = 0
          if len(food_dist) > 0:
            nearest_coin = min(food_dist)

            if nearest_coin < 3:
              nearest_coin = 9
            else:
              nearest_coin = 0

          score += very_near_ghost
          score += nearest_ghost
          return score

        # In order to calculate the better state, we calcualte the difference between the current
        # and the new states
        return stateValue(currentGameState.generateSuccessor(0, action)) - stateValue(currentGameState)


def manhattanDistance( xy1, xy2 ):
    "Returns the Manhattan distance between points xy1 and xy2"
    return abs( xy1[0] - xy2[0] ) + abs( xy1[1] - xy2[1] )

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class TreeNode:
  def __init__(self):
    self.successsors = []

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        def minimaxSearch(node, depth, agent):
          # Dynamic programing optimization

          act_list = node.getLegalActions(agent)
          # Terminal node
          if depth == 0 or act_list == [] or node.isWin() or node.isLose():
            return self.evaluationFunction(node)

          # Increment the agent
          next_agent = agent + 1
          if next_agent == node.getNumAgents():
            next_agent = 0
            # If we restart the agent count, we go down in the depth of the search
            new_depth = depth - 1
          else:
            new_depth = depth

          state_list = [(node.generateSuccessor(agent, act), act) for act in act_list]
          score_list = [ minimaxSearch(state, new_depth, next_agent) for state, act in state_list ]

          if agent == 0: # Max meassure
            result = max(score_list)
          else:
            result = min(score_list)

          return result

        # Calcualte the best action in all the alternatives
        total_act = [(minimaxSearch(gameState.generateSuccessor(0, act), self.depth, 1), act) for act in gameState.getLegalActions(0)]
        return max(total_act)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        def ABSearch(node, depth, alpha, beta, agent):
          # Dynamic programing optimization

          act_list = node.getLegalActions(agent)
          # Terminal node
          if depth == 0 or act_list == [] or node.isWin() or node.isLose():
            return self.evaluationFunction(node)

          # Increment the agent
          next_agent = agent + 1
          if next_agent == node.getNumAgents():
            next_agent = 0
            new_depth = depth - 1
          else:
            new_depth = depth

          if agent == 0: # Max meassure
            result = -9999
            for act in act_list:
              state = node.generateSuccessor(agent, act)
              result = max(result, ABSearch(state, new_depth, alpha, beta, next_agent))
              if result > beta:
                return result # Prunning
              alpha = max(result, alpha)
              
          else:
            result = 9999
            for act in act_list:
              state = node.generateSuccessor(agent, act)
              result = min(result, ABSearch(state, new_depth, alpha, beta, next_agent))
              if result < alpha:
                return result # Prunning
              beta = min(result, beta)

          return result

        total_act = [(ABSearch(gameState.generateSuccessor(0, act), self.depth, -9999, 9999, 1), act) for act in gameState.getLegalActions(0)]
        return max(total_act)[1]
        # Alternative implementation (one of many):
        '''
        def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        def ABPrunningSearch(node, depth, alpha, beta, agent):         
          # Terminal node
          if depth == 0:
            return (self.evaluationFunction(node), None)

          act_list = node.getLegalActions(agent)
          if act_list == []:
              return (self.evaluationFunction(node), None)

          next_agent = agent + 1
          if next_agent == node.getNumAgents():
            next_agent = 0
            new_depth = depth - 1
          else:
            new_depth = depth

          state_list = [(node.generateSuccessor(agent, act), act) for act in act_list]

          for state, act in state_list:
            score = ABPrunningSearch(state, new_depth, alpha, beta, next_agent)
          
            if agent == 0:
              alpha = max(score, alpha)
            else:
              beta = min(score, beta)

            if beta <= alpha:
              break

          if agent == 0: 
            print(alpha, beta)
            return alpha
          else:
            return beta

          return ABPrunningSearch(gameState, self.depth, -9999, +9999, 0)
          '''

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        from random import seed, randint, random

        def expectimax(node, depth, agent):
          act_list = node.getLegalActions(agent)
          # Terminal node
          if depth == 0 or act_list == []:
            return self.evaluationFunction(node)

          # Increment the agent
          next_agent = agent + 1
          if next_agent == node.getNumAgents():
            next_agent = 0
            new_depth = depth - 1
          else:
            new_depth = depth

          state_list = [(node.generateSuccessor(agent, act), act) for act in act_list]
          score_list = [ expectimax(state, new_depth, next_agent) for state, act in state_list ]

          if agent == 0: # Max meassure
            result = max(score_list)
          else:
            # If is the turn of the advesarial agent, we calcualte the score according to a 
            # uniform distribution
            result = .0
            for score in score_list:
              result += score
            result *= 1./len(score_list)
          return result

        total_act = [(expectimax(gameState.generateSuccessor(0, act), self.depth, 1), act) for act in gameState.getLegalActions(0)]
        
        return max(total_act)[1]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION:
      I tried multiple aproaches, that I included bellow, in order to find the best aproach
      
      The best aproach, with 10 wins, 1088.9 of score and 6/6 os score is made
      with the intent to to measure the distance to the closest elements.
      Since I stablished the metrics as a penalization or bonifiaction for closeness,
      I substracted the diference to a constant in order to obtain a "negative" of the
      distance, so it would be easier to penalize.
      I also omited the ghosts from teh search if they are in a afraid state.
      After that it would be trial and error in order to get the correct weights.
    """
    # 6/6 10 wins score 1088.9
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    ghost_positions = [g_state.getPosition()  for g_state in currentGameState.getGhostStates()]
    capsList = currentGameState.getCapsules()

    # Calculate all the distances to pacman
    food_dist  = [manhattanDistance(newPos, x) for x in newFood.asList()]
    # Pack all the ghost information
    ghost_compress = zip(ghost_positions, newScaredTimes)
    # Ommit all the ghosts that are afraid from the search
    ghost_dist = [manhattanDistance(newPos, x) for x, y in ghost_compress if y == 0.]

    caps_dist = [manhattanDistance(newPos, x) for x in capsList]

    # Calculate the "negative" distance in order for it to be easier to penalize
    # Also divided by half for it to be easier to scale it up
    if ghost_dist != []: # For avoid a calculation of a empty array
      nearest_ghost = abs(100 - min(ghost_dist)) / 2.
    else:
      nearest_ghost = 0.
    if food_dist != []:
      nearest_coin = abs(100 - min(food_dist))/ 2.
    else:
      nearest_coin = 0.
    if caps_dist != []:
      nearest_capsules = abs(100 - min(caps_dist))/ 2.
    else:
      nearest_capsules = 0.

    # Weight calculation
    score = currentGameState.getScore()
    score += nearest_ghost * - 2
    score += nearest_coin * 2.5
    score += nearest_capsules * 2
    
    return score
    # 5/6 10 wins score 939.0
    # Same as the previus version but without taking into account to ghost scare time, and with
    # only a little of fine tunning the weights 
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    ghost_positions = [g_state.getPosition()  for g_state in currentGameState.getGhostStates()]
    capsList = currentGameState.getCapsules()

    food_dist  = [manhattanDistance(newPos, x) for x in newFood.asList()]
    ghost_dist = [manhattanDistance(newPos, x) for x in ghost_positions]
    caps_dist = [manhattanDistance(newPos, x) for x in capsList]

    if ghost_dist != []:
      nearest_ghost = abs(100 - min(ghost_dist)) / 2.
    else:
      nearest_ghost = 0.
    if food_dist != []:
      nearest_coin = abs(100 - min(food_dist))/ 2.
    else:
      nearest_coin = 0.
    if caps_dist != []:
      nearest_capsules = abs(100 - min(caps_dist))/ 2.
    else:
      nearest_capsules = 0.

    score = currentGameState.getScore()
    score += nearest_ghost * -3
    score += nearest_coin * 2.5
    score += nearest_capsules * 2
    return score
    # Previous version
    # 4/6 9 wins score 789.3
    # In this version, I computed not only the nearest points, but the elements in proximity
    # of pacman, via a combination of the number of elements, and the sum of then
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    ghost_positions = [g_state.getPosition()  for g_state in currentGameState.getGhostStates()]

    # Calculate distances
    food_dist  = [manhattanDistance(newPos, x) for x in newFood.asList()]
    ghost_dist = [manhattanDistance(newPos, x) for x in ghost_positions]
    capsList = [manhattanDistance(newPos, x) for x in currentGameState.getCapsules()]

    near_ghost = len([ x for x in ghost_dist if x > 2 and x < 3])
    very_near_ghost = sum([ x for x in ghost_dist if x < 2])
    nearest_coins = len([ x  for x in food_dist if x < 2])
    nearest_capsules = len([ x  for x in capsList if x < 5])

    score = currentGameState.getScore()/2.
    score += very_near_ghost * -5
    score += nearest_coins * 3
    score += nearest_capsules * 4
    return score


# Abbreviation
better = betterEvaluationFunction

