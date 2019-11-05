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
        print("food", newFood.asList())
        print("curr_pos ", newPos)
        print("Ghost_state ", ghost_positions)
        print("new scare time ", newScaredTimes)

        food_dist  = [manhattanDistance(newPos, x) for x in newFood.asList()]
        
        food_dist.sort()

        print(food_dist)

        def ghost_mapping_func(ghost_pos, scare_time):
          ghost_score = 0
          dist = manhattanDistance(newPos, ghost_pos)
          if scare_time > dist:
            return 0
          if dist <= 4.0:
            return -dist * 5
          return dist

        ghost_states = zip(ghost_positions, newScaredTimes)

        ghost_dist  = [ghost_mapping_func(pos, scare) for pos, scare in ghost_states]

        food_total_nearest_distance = sum(food_dist[:len(ghost_dist)])
        food_total_furthest_distance = sum(food_dist[(len(food_dist) - len(ghost_dist)) :])
        ghost_total_distance = min(ghost_dist)

        #score = successorGameState.getScore() + food_total_nearest_distance * 0.5 + food_total_furthest_distance *0.5  + ghost_total_distance
        score = successorGameState.getScore() * 0.6 + (sum(food_dist)/len(food_dist)) * 1.5 + ghost_total_distance + food_total_nearest_distance


        print("score ", score)
        print("avg sfood", (sum(food_dist)/len(food_dist)))
        print("food near dist", food_total_nearest_distance)
        print("food furthers dist", food_total_furthest_distance)
        print("ghost total distance ", ghost_total_distance)
        print(successorGameState.getScore())

        "*** YOUR CODE HERE ***"
        #return successorGameState.getScore()
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
        stack = [(None, gameState, 0)]

        agents = range(gameState.getNumAgents())
        agent_id = 0

        curr_depth = 0

        tree = {}
        node_value = {}

        # Generate Game Tree
        while len(stack) > 0:
          package_state = stack.pop()
          action_state, curr_game_state, state_depth = package_state
          act_list = curr_game_state.getLegalActions(agents[agent_id])

          state_list = [(curr_game_state.generateSuccessor(agents[agent_id], act), act) for act in act_list]

          tree[curr_game_state] = state_list

          # If we reached the maximun depth, we stop adding new elements to the stack
          if self.depth > state_depth and state_list != []:
            for new_state, action_to_state in state_list:
              stack.append((action_to_state, new_state, state_depth + 1))

          else:
            # Add the nodes as leafs to the tree
            for new_state, action_to_state in state_list:
              tree[new_state] = []

          # Switching the agents, to calculate the game tree
          agent_id += 1
          if agent_id == gameState.getNumAgents():
            agent_id = 0

        result = getBestAction(gameState, 0, tree)

        return result[1]

def getBestAction(node, agent_id, tree):
  if tree[node] == []:
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
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

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
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

