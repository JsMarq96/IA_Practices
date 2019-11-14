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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        ghost_positions = [g_state.getPosition()  for g_state in successorGameState.getGhostStates()]


        food_dist  = [manhattanDistance(newPos, x) for x in newFood.asList()]
        
        #food_dist.sort()

        #food_dist[len(food_dist)] - food_dist[0]
        if len(food_dist) == 0:
          fd = 0.01
        else:
          #fd = (food_dist[len(food_dist)-1] - food_dist[0]) * 0.50
          fd = min(food_dist)

        ghost_dist = [manhattanDistance(newPos, x) for x in ghost_positions]

        # + (fd) * 0.25
        # (len(newScaredTimes) * 0.3)

        score = -len(newFood.asList()) #* -0.3 #+ (sum(ghost_dist))

        # Emergency Ghost
        # Socre 424.8 no fails
        very_near_ghost = sum([ -99  for x in ghost_dist if x < 2])

        # score 188.2 no fails
        # ghost_stats = zip(ghost_dist, newScaredTimes)
        # very_near_ghost = sum([ -999  for x, y in ghost_stats if x < 2 and y <= 2])

        # score 154 4 loses 1/4
        #very_near_ghost = sum([ -999  for x, y in ghost_stats if x < 4 and y <= 2])

        # 408 2 fails
        # very_near_ghost = sum([ -999  for x, y in ghost_stats if x < 3 and y <= 5])


        nearest_ghost = min(ghost_dist)

        '''
        # Score 204.5 No fail
        if nearest_ghost > 6:
          nearest_ghost = 999
        else:
          nearest_ghost = 0
        '''

        # Score 507.7 No fail
        if nearest_ghost >= 6:
          nearest_ghost = 999
        else:
          nearest_ghost = 0

        nearest_coin = 0
        if len(food_dist) > 0:
          nearest_coin = min(food_dist)

          if nearest_coin < 3:
            nearest_coin = 9
          else:
            nearest_coin = 0

        score += very_near_ghost #* 1.6
        score += nearest_ghost
        #score += nearest_coin

        #print(very_near_ghost, nearest_ghost, min(food_dist), food_dist)
        #score = very_near_ghost + fd * 999.

        return score


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
          score_list = [ minimaxSearch(state, new_depth, next_agent) for state, act in state_list ]

          if agent == 0: # Max meassure
            result = max(score_list)
          else:
            result = min(score_list)

          return result

        total_act = [(minimaxSearch(gameState.generateSuccessor(0, act), self.depth, 1), act) for act in gameState.getLegalActions(0)]
        return max(total_act)[1]
        return minimaxSearch(gameState, self.depth, 0)[1]

def getBestAction(node, agent_id, tree):
  if tree[node] == []:
    print("term node", node, node.getScore())
    return node.getScore(), None

  new_agent_id = agent_id + 1
  if new_agent_id == node.getNumAgents():
    new_agent_id = 0

  score_list = [ (getBestAction(it_node, new_agent_id, tree)[0], node_act) for it_node, node_act in tree[node] ]

  if agent_id == 0: # If the depth is even, then it's a Max choice
    #print('max', max(score_list), agent_id, score_list)
    result =  max(score_list)
  else:
    #print('min', min(score_list), agent_id, score_list)
    result = min(score_list)

  return result[0], result[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        """
        def ABSearch(node, depth, alpha, beta, agent):
          # Dynamic programing optimization

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

          if agent == 0: # Max meassure
            result = -9999
            for act in act_list:
              state = node.generateSuccessor(agent, act)
              result = max(result, ABSearch(state, new_depth, alpha, beta, next_agent))
              if result > beta:
                return result
              alpha = max(result, alpha)
              
          else:
            result = 9999
            for act in act_list:
              state = node.generateSuccessor(agent, act)
              result = min(result, ABSearch(state, new_depth, alpha, beta, next_agent))
              if result < alpha:
                return result
              beta = min(result, beta)

          return result

        total_act = [(ABSearch(gameState.generateSuccessor(0, act), self.depth, -9999, 9999, 1), act) for act in gameState.getLegalActions(0)]
        return max(total_act)[1]

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
            result = .0
            #t = 1./randint(1, len(score_list))
            t = 1./len(score_list)
            #t = 1./5
            #t = random()
            for score in score_list:
              result += score * 1./len(score_list)
              #result += score * t
          return result

        total_act = [(expectimax(gameState.generateSuccessor(0, act), self.depth, 1), act) for act in gameState.getLegalActions(0)]
        
        return max(total_act)[1]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    # 3/6 8 wins score -239
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    ghost_positions = [g_state.getPosition()  for g_state in currentGameState.getGhostStates()]


    food_dist  = [manhattanDistance(newPos, x) for x in newFood.asList()]

    if len(food_dist) == 0:
      fd = 0.01
    else:
      fd = min(food_dist)

    ghost_dist = [manhattanDistance(newPos, x) for x in ghost_positions]

    very_near_ghost = sum([ -99  for x in ghost_dist if x < 2])
    nearest_ghost = min(ghost_dist)
    if nearest_ghost >= 6:
      nearest_ghost = 999
    else:
      nearest_ghost = 0

    nearest_coin = 0
    if len(food_dist) > 0:
      nearest_coin = min(food_dist)

      if nearest_coin < 3:
        nearest_coin = 9
      else:
        nearest_coin = 0

    score = -len(newFood.asList()) #* -0.3 #+ (sum(ghost_dist))
    score += very_near_ghost #* 1.6
    score += nearest_ghost
    #score += nearest_coin

    return score
    # 3/6 8 wins score -239
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    ghost_positions = [g_state.getPosition()  for g_state in newGhostStates]


    food_dist  = [manhattanDistance(newPos, x) for x in newFood.asList()]

    ghost_dist = [manhattanDistance(newPos, x) for x in ghost_positions]

    #very_near_ghost = sum([ -99  for x in ghost_dist if x <= 2])

    ghost_stats = zip(ghost_dist, newScaredTimes)
    very_near_ghost = sum([ -99  for x, y in ghost_stats if x < 3 and y <= 3])
    
    nearest_ghost = min(ghost_dist)
    if nearest_ghost >= 6:
      nearest_ghost = 999
    else:
      nearest_ghost = 0

    nearest_coin = 0
    if len(food_dist) > 0:
      fd = sum([ 99  for x in food_dist if x <= 2])
    else:
      fd = 0.

    score = -len(newFood.asList()) #* -0.3 #+ (sum(ghost_dist))
    score += very_near_ghost * .6
    score += nearest_ghost
    score += fd

    return score

# Abbreviation
better = betterEvaluationFunction

