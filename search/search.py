# search.py
# ---------
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
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def searchStruct(problem, start_state, dataStruct_isEmpty, dataStruct_Add, dataStruct_Pop):
    """
    Container of the default behaviour of the search problem, but with a abstraction
    in the data strucutre part, in order to rehutilize this code in the multiple search
    algortitmhs, like DFS and BFS.
    The parameters are the problem, the start state and the actions of the data struct.
    """
    from game import Directions
    goal_state = None
    visited = set()

    dataStruct_Add( (None, start_state, None) )
    visited.add(start_state)

    # For parsing the directions outputed in GetSuccessors, to actual directions
    dir_translator = {'South' : Directions.SOUTH , 'West' : Directions.WEST , 'East' : Directions.EAST, 'North': Directions.NORTH,
    'up' : Directions.NORTH, 'left' : Directions.WEST, 'right' : Directions.EAST, 'down' : Directions.SOUTH}

    while not dataStruct_isEmpty():
        curr_node = dataStruct_Pop()
        _, c_node, _ = curr_node

        if problem.isGoalState(c_node):
            goal_state = curr_node
            break
        else:
            child_nodes = problem.getSuccessors(c_node)

            for it_node in child_nodes:
                child = it_node[0]
                child_action = it_node[1]
                # Visits unvisited node if necessary
                if child not in visited:
                    dataStruct_Add((curr_node, child, child_action))
                    visited.add(child)

    back_trace = []
    iterator = goal_state

    # Unpack the goal_node hysory, parent by parent
    while iterator[0] != None:
        father, _, action = iterator
        back_trace.insert(0, dir_translator[action])
        iterator = father

    return back_trace

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Goal State:", problem.goal
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState()) """ 
    stack = util.Stack()

    start_state = problem.getStartState()
    return searchStruct(problem, start_state, stack.isEmpty, stack.push, stack.pop)

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    queue = util.Queue()
    start_state = problem.getStartState()
    return searchStruct(problem, start_state, queue.isEmpty, queue.push, queue.pop)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    from game import Directions
    queue = util.PriorityQueue()
    start_state = problem.getStartState()
    goal_state = None
    visited = set()

    queue.push( (None, start_state, None), 0)
    visited.add(start_state)

    # For parsing the directions outputed in GetSuccessors, to actual directions
    dir_translator = {'South' : Directions.SOUTH , 'West' : Directions.WEST , 'East' : Directions.EAST, 'North': Directions.NORTH,
    'up' : Directions.NORTH, 'left' : Directions.WEST, 'right' : Directions.EAST, 'down' : Directions.SOUTH}

    while not queue.isEmpty():
        curr_node = queue.pop()
        _, c_node, _ = curr_node

        if problem.isGoalState(c_node):
            goal_state = curr_node
            break
        else:
            child_nodes = problem.getSuccessors(c_node)

            for it_node in child_nodes:
                child = it_node[0]
                child_action = it_node[1]
                # Visits unvisited node if necessary
                if child not in visited:
                    queue.push((curr_node, child, child_action), it_node[2])
                    visited.add(child)

    back_trace = []
    iterator = goal_state

    # Unpack the goal_node hysory, parent by parent
    while iterator[0] != None:
        father, _, action = iterator
        back_trace.insert(0, dir_translator[action])
        iterator = father

    return back_trace

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
