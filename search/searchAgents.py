# searchAgents.py
# ---------------
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


"""
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""

from game import Directions
from game import Agent
from game import Actions
import util
import time
import search

class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError, fn + ' is not a search function in search.py.'
        func = getattr(search, fn)
        if 'heuristic' not in func.func_code.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError, heuristic + ' is not a function in searchAgents.py or search.py.'
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError, prob + ' is not a search problem type in SearchAgents.py.'
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception, "No search function provided for SearchAgent"
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print 'Warning: this does not look like a regular search maze'

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn, (1, 1), None, False)

class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height-2, self.walls.width-2
        self.corners = ((1,1), (1,top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print 'Warning: no food in corner ' + str(corner)
        self._expanded = 0 # DO NOT CHANGE; Number of search nodes expanded
        # Please add any code here which you would like to use
        # in initializing the problem
        
        ''' State structure = (Visited set, current position) '''        
        self.corners_list = [ self.corners[0], self.corners[1], self.corners[2] ,self.corners[3] ]
        self.start_state = (frozenset(), self.startingPosition)
        self.goal_state = (frozenset(self.corners_list), None)


    def getStartState(self):
        """
        Returns the start state (in your state space, not the full Pacman state
        space)
        """
        return self.start_state

    def isGoalState(self, state):
        """
        Returns whether this search state is a goal state of the problem.
        """
        return state[0] == self.goal_state[0]

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
            For a given state, this should return a list of triples, (successor,
            action, stepCost), where 'successor' is a successor to the current
            state, 'action' is the action required to get there, and 'stepCost'
            is the incremental cost of expanding to that successor
        """
        visited_corners = state[0]
        curr_x, curr_y = state[1]

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(curr_x + dx), int(curr_y + dy)

            if not self.walls[nextx][nexty]:
                new_pos = (nextx, nexty)

                curr_set = set(visited_corners)
                if new_pos in self.corners and new_pos not in curr_set:
                    curr_set.add(new_pos)

                successors.append( ((frozenset(curr_set), new_pos), action, 1) )

        self._expanded += 1 # DO NOT CHANGE
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None: return 999999
        x,y= self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)

def euclideanHeuristicToPoint(point_1, point_2):
    '''
    Euclidean distance heuristic from point 1 to point 2
    Note: Based on the Euclidean Heurisitic provided
    '''
    return ( (point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2 ) ** 0.5

def numberOfWalls(point_1, point_2, walls):
    '''
    Heurisitic function that counts the number of walls in a straight line from
    point 1 to point 2.
    It was an attempt to determine the complexity and add it  to a distance
    NOTE: It has incosistency problems (by design)
    '''
    n_steps = 0
    n_walls = 0

    def mapToLookup(v1, v2):
        '''
        Helper function that translates the direction from the vector v1-v2
        to a unit vectro to move in a grid
        '''
        import math

        degrees = math.degrees(math.atan2(v1, v2))

        if degrees < 0:
            degrees += 360

        index = int(round(degrees / 45))
        directions = [(0,1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1) ]
        
        return directions[index]

    it_dir_x = int(point_1[0])
    it_dir_y = int(point_1[1])

    while it_dir_x != point_2[0] and it_dir_y != point_2[1]:
        # If there is a wall, increment the counter
        if not walls[it_dir_x][it_dir_y]:
            n_walls += 1

        # Advance one spet in the direccion curr_point-goal (point_2)
        dir_x, dir_y = mapToLookup(point_2[0] - it_dir_x, point_2[1] - it_dir_y)

        it_dir_x += dir_x
        it_dir_y += dir_y

        n_steps += 1

    return n_walls

def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return a number that is a lower bound on the
    shortest path from the state to a goal of the problem; i.e.  it should be
    admissible (as well as consistent).
    """
    corners = problem.corners # These are the corner coordinates
    walls = problem.walls # These are the walls of the maze, as a Grid (game.py)

    'Different Heuristic Functions tested:'

    'Expended nodes on Medium Size: 806 (Euclidean Distance)'
    #metric_func = euclideanHeuristicToPoint

    'Expended nodes on Medium Size: 692 (Manhattan Distance) (Best result)'
    metric_func = util.manhattanDistance

    'Expended nodes on Medium Size: 1529 (Number of walls)'
    #metric_func = lambda start, state: numberOfWalls(start, state, walls)

    # NOTE: We also experimented on convining the different Heuristics

    'Expended nodes on Medium Size: 1323 (number of walls + Manhatan Distance)'
    #metric_func = lambda start, state: numberOfWalls(start, state, walls) * .5 + util.manhattanDistance(start, state) * .5
    
    'Expended nodes on Medium Size: 1145 (number of walls + euclidiean distance)'
    #metric_func = lambda start, state: numberOfWalls(start, state, walls) * .4 + euclideanHeuristicToPoint(start, state) * .4
    
    'Expended nodes on Medium Size: 743 (Manhattan distance + Euclidiean Distance)'
    #metric_func = lambda start, state: util.manhattanDistance(start, state) * .5 + euclideanHeuristicToPoint(start, state) * .5


    unvisited_corners = [x for x in corners if not x in state[0]]

    total_heuristic = 0.

    start_state = state[1]
    while len(unvisited_corners) != 0:

        heu, start_state = min([(metric_func(start_state, state), state) for state in unvisited_corners])

        unvisited_corners.remove(start_state)

        total_heuristic += heu

    return total_heuristic # Default to trivial solution

class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem

class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0 # DO NOT CHANGE
        self.heuristicInfo = {} # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1 # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem

class CoinCluster:
    '''
    Class that manages Coin clusters in a cleaner way
    '''
    def __init__(self, walls):
        self.element_list = []
        self.num_walls = 0
        self.walls = walls

    '''HELPER FUNCTIONS'''
    def __iter__(self):
        return self

    def __getitem__(self, pos):
        return self.element_list[pos]

    def __len__(self):
        return len(self.element_list)

    def __contains__(self, item):
        return item in self.element_list

    def __iadd__(self, new_pos):
        ''' Add a element to the cluster '''
        if not new_pos in self.element_list:
            self.element_list.append(new_pos)
            self.num_walls += len(getWallsArroundPos(new_pos, self.walls))
        return self

    def __isub__(self, other):
        ''' Removes a element from the cluster '''
        if other in self.element_list:
            self.element_list.remove(other)
            self.num_walls -= len(getWallsArroundPos(other, self.walls))
        return self

    ''' LOGIC FUNCTIONS '''
    def isPartOfTheCluster(self, new_pos, search_range = 1.0):
        ''' With a given range, returns if the given coordinates is a neighboor to the cluster '''
        count = len([item for item in self.element_list if euclideanHeuristicToPoint(new_pos, item) == search_range])
        return count > 0

    def getFurthestInCluster(self, reference, metric = util.manhattanDistance):
        ''' With a given point, get the element in the cluster that is furthest away '''
        return max([ (metric(reference, item), item) for item in self.element_list ])

    def getNumOfNeighbooringWalls(self):
        ''' Return the number of walls of the Cluster '''
        return self.num_walls

    def getScore(self):
        ''' Returns the score obtaibned in the cluster '''
        return len(self.element_list)

def getWallsArroundPos(pos, walls):
    '''
    Utility function to count walls arround a position.
        It look up the neighbooring positions
    '''
    p_x , p_y = pos
    walls_list = []
    for p_wall_x in [1, -1]:
        walls_list += [(p_x + p_wall_x, p_y + p_wall_y) for p_wall_y in [1, -1] if walls[p_x + p_wall_x][p_y + p_wall_y]]
    return walls_list

def coinClusteringHeuristic(position, foodGrid, problem):
    '''
        Heuristic function that Focuses on finding the nearest cluster of coins:
         1) Fisrt it clusters the coins based on the remain coins (Optimize this)
         2) For each cluster, we get the direction of the furthest point in the cluster
            if that distance, select the minuum of all
         3) We do this until we visit every cluster, and calculate the cost

         Performance: 12517 2/4
    '''
    cluster_list = []
    walls = problem.walls

    for food in foodGrid.asList():
        for cluster in cluster_list:
            # Check if the current item is part of any cluster:
            #  If the distance between any of the cluster elements and the current food is 1, is part
            #  of the cluster
            if cluster.isPartOfTheCluster(food):
                cluster += food
                break
        else:
            # If it goes thought all the clusters, and istn near everyone, create new cluster
            new_cluster = CoinCluster(walls)
            new_cluster += food
            cluster_list.append(new_cluster)


    start_pos = position
    heuristic = 0

    while len(cluster_list) > 0:
        min_dist = 9999.
        min_cluster = None
        min_pos = None

        for cluster in cluster_list:
            # Get farthest point of the cluster from the current position
            cost, curr_pos = cluster.getFurthestInCluster(start_pos)
            cost += cluster.getNumOfNeighbooringWalls() * 0.5

            # Select the cluster if its less than the curretn minimun distance
            if cost < min_dist:
                min_dist = cost
                min_cluster = cluster
                min_pos = curr_pos

        # Remove cluster and update the position
        cluster_list.remove(min_cluster)
        start_pos = min_pos

        heuristic += min_cluster.getScore()
        #most_val, most_val_cluster = max([ (len(clust), clust) for clust in cluster_list ])

    return heuristic

def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state

    # NOTE: There where plans of adding a heuristic based on the number of walls surrounding
    # the clusters, but due to inconsistency reasons, it was discarded

    return coinClusteringHeuristic(position, foodGrid, problem)
   

class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"
    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while(currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception, 'findPathToClosestDot returned an illegal move: %s!\n%s' % t
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print 'Path found with cost %d.' % len(self.actions)

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)
                
        # Use breath First Search to locate the nearest Dot
        return search.breadthFirstSearch(problem)

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state

        if self.food[x][ y]:
            return True
        else:
            return False

def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))
