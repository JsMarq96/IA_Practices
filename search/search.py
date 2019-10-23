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

def searchStruct1(problem, start_state, dataStruct_isEmpty, dataStruct_Add, dataStruct_Pop, debug = False, dStrct = None):
    """
    Container of the default behaviour of the search problem, but with a abstraction
    in the data strucutre part, in order to rehutilize this code in the multiple search
    algortitmhs, like DFS and BFS.
    The parameters are the problem, the start state and the actions of the data struct.
    """
    from game import Directions
    goal_route = None
    visited = set()

    dataStruct_Add( (start_state, []) )
    visited.add(start_state)

    while not dataStruct_isEmpty():
        curr_node = dataStruct_Pop()
        c_node, act_list = curr_node

        if debug:
            print(curr_node, dStrct)
        if problem.isGoalState(c_node):
            return act_list
    

        for child, child_action, cost in problem.getSuccessors(c_node):
            # Visits unvisited node if necessary
            if debug:
                print(child, child not in visited)
            if child not in visited:
                child_list = act_list + [ child_action ]
                if debug:
                    print(child_list)
                dataStruct_Add((child, child_list))

        visited.add(c_node)

    print(goal_route)
    #print('*********************')
    return None

def searchStruct(problem, start_state, dataStruct_isEmpty, dataStruct_Add, dataStruct_Pop):
    """
    Container of the default behaviour of the search problem, but with a abstraction
    in the data strucutre part, in order to rehutilize this code in the multiple search
    algortitmhs, like DFS and BFS.
    The parameters are the problem, the start state and the actions of the data struct.
    """
    from game import Directions
    goal_route = None
    visited = set()

    dataStruct_Add( (start_state, []) , 0)

    while not dataStruct_isEmpty():
        c_node, act_list = dataStruct_Pop()

        if problem.isGoalState(c_node):
            return act_list

        if c_node not in visited:
            for child, child_action, cost in problem.getSuccessors(c_node):
                # Visits unvisited node if necessary
                if child not in visited:
                    dataStruct_Add((child, act_list + [ child_action ]), cost)

            visited.add(c_node)

    print(goal_route)
    return None

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first. Returns the actions list.
    Uses the searchStruct function with the Stack functions.

    print "Start:", problem.getStartState()
    print "Goal State:", problem.goal
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState()) """
    stack = util.Stack()

    start_state = problem.getStartState()

    stackPush = lambda x, cost: stack.push(x)

    return searchStruct(problem, start_state, stack.isEmpty, stackPush, stack.pop)

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    Uses the searchStruct function with the Queue functions.
    """

    queue = util.Queue()
    start_state = problem.getStartState()

    queuePush = lambda x, cost: queue.push(x)

    return searchStruct(problem, start_state, queue.isEmpty, queuePush, queue.pop)

def uniformCostSearch(problem):
    """Search the node of least total cost first.
    from game import Directions
    """
    ''' VER 1.0  -> PETAN TODOS MENOS UNO'''
    queue = util.PriorityQueue()
    start_state = problem.getStartState()

    #queuePush = lambda x, cost: queue.push(x, cost)

    return searchStruct(problem, start_state, queue.isEmpty,  queue.push, queue.pop)
    ''' VER 2.0  -> PETAN 2 PERO INCOROPORA REORDENACION'''
    queue = util.PriorityQueue()
    start_state = problem.getStartState()
    goal_route = None
    visited = set()
    frontier_dict = { start_state : None}

    queue.push( (start_state, []) , 0)

    while not queue.isEmpty():
        # Remove Value from the Queue, and from the frontier
        c_node, act_list = queue.pop()
        del frontier_dict[c_node]

        if problem.isGoalState(c_node):
            return act_list

        if c_node not in visited:
            for child, child_action, cost in problem.getSuccessors(c_node):
                
                if child not in frontier_dict: # If the child have not been visited, queue it
                    new_node = (child, act_list + [ child_action ])
                    frontier_dict[child] = new_node
                    queue.push(new_node, cost)
                else: # If Its in the queue, we update the cost
                    node_to_update = frontier_dict[child]
                    queue.update(node_to_update, cost)

            visited.add(c_node)

    print(goal_route)
    return None

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    queue = PriorityQueueWithFunction()
    start_state = problem.getStartState()


    return searchStruct(problem, start_state, queue.isEmpty,  queue.push, queue.pop)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
