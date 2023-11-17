import numpy as np
from state import next_state, solved_state
from location import next_location
from collections import OrderedDict
import heapq
import time
from location import solved_location


def solve(init_state, init_location, method):
    """
    Solves the given Rubik's cube using the selected search algorithm.

    Args:
        init_state (numpy.array): Initial state of the Rubik's cube.
        init_location (numpy.array): Initial location of the little cubes.
        method (str): Name of the search algorithm.

    Returns:
        list: The sequence of actions needed to solve the Rubik's cube.
    """

    class Node:
        cube = None
        cost = 0
        parent = None
        move = None
        heuristic = None
        location = None

        def __lt__(self, other):
            # Python's heapq module requires items in the heap to be comparable.
            # so that Python can properly order the items in the priority queue.
            # Less than comparison method
            # For A* algorithm, compare nodes based on their total cost (cost + heuristic)
            return (self.cost + self.heuristic) < (other.cost + other.heuristic)

    def heuristic(current_location):
        goal_state = solved_location()

        distance = 0
        current_location = np.array(current_location)

        # It iterates through the eight cubes of the 2x2 Rubik's Cube (range(8)).
        for i in range(8):
            for layer in range(2):
                for col in range(2):
                    for row in range(2):
                        # Compares the current cube's position with its corresponding position in the solved state.
                        if (
                            current_location[row, col, layer]
                            != goal_state[row, col, layer]
                        ):
                            # For a misplaced cube (current_location not matching goal_state), it finds the cube's position in the goal state.
                            # Calculates the Manhattan distance between the misplaced cube's current position and its ideal position in the solved state.

                            for layer_ in range(2):
                                for col_ in range(2):
                                    for row_ in range(2):
                                        if (
                                            current_location[row_, col_, layer_]
                                            == goal_state[row, col, layer]
                                        ):
                                            # The Manhattan distance is computed by summing the absolute differences between the row, column, and layer values.
                                            distance += (
                                                np.abs(row_ - row)
                                                + np.abs(col_ - col)
                                                + np.abs(layer_ - layer)
                                            )

            return distance / 4

    if method == "Random":
        return list(np.random.randint(1, 12 + 1, 10))

    elif method == "IDS-DFS":
        cost_limit = 1
        nodes_explored = 0
        nodes_expanded = 0
        depth_of_answer = 0
        moves = list()
        frontier = OrderedDict()
        solved = solved_state()

        start_time = time.time()

        while True:
            start = Node()
            start.cube = init_state
            frontier[hash(start.cube.tobytes())] = start

            while len(frontier) != 0:
                _, curr = frontier.popitem(last=False)
                nodes_explored += 1

                if (curr.cube == solved).all():
                    node_search = curr
                    while node_search.parent is not None:
                        moves.append(node_search.move)
                        node_search = node_search.parent
                    moves.reverse()
                    depth_of_answer = len(moves)
                    execution_time = time.time() - start_time

                    print("Nodes Explored:", nodes_explored)
                    print("Nodes Expanded:", nodes_expanded)
                    print("Depth of Answer:", depth_of_answer)
                    print("Time Taken:", execution_time)

                    return moves

                if curr.cost + 1 <= cost_limit:
                    child_cost = curr.cost + 1
                    for i in range(12):
                        nodes_expanded += 1
                        new = Node()
                        new.cube = next_state(curr.cube, i + 1)
                        new.cost = child_cost
                        new.parent = curr
                        new.move = i + 1

                        # Check for duplicate states in the path
                        duplicate_state = False
                        node_search = curr
                        while node_search.parent is not None:
                            if (node_search.parent.cube == new.cube).all():
                                duplicate_state = True
                                break
                            node_search = node_search.parent

                        if not duplicate_state:
                            frontier[hash(new.cube.tobytes())] = new

            cost_limit += 1

        return list(np.random.randint(1, 12 + 1, 10))

    elif method == "A_star":
        start_time = time.time()

        # Initialize variables
        cost_limit = 1
        nodes_explored = 0
        nodes_expanded = 0
        depth_of_answer = 0
        moves = list()
        frontier = []  # Use a list for the priority queue
        solved = solved_state()

        start = Node()
        start.cube = init_state
        start.cost = 0
        start.location = init_location

        # Heuristic for the start node
        start.heuristic = heuristic(init_location)

        # Priority queue in A*
        heapq.heappush(frontier, (start.cost + start.heuristic, start))
        visited = set()

        # while frontier is not empty
        while frontier:
            _, curr = heapq.heappop(frontier)

            # If already visited
            if hash(curr.cube.tobytes()) in visited:
                continue

            # Mark the current state as visited
            visited.add(hash(curr.cube.tobytes()))

            nodes_explored += 1

            if hash(curr.cube.tobytes()) == hash(solved.tobytes()):
                # Goal state reached
                node_search = curr
                while node_search.move is not None:
                    moves.append(node_search.move)
                    node_search = node_search.parent
                moves.reverse()
                depth_of_answer = len(moves)
                end_time = time.time()
                print("Nodes Explored:", nodes_explored)
                print("Nodes Expanded:", nodes_expanded)
                print("Depth of Answer:", depth_of_answer)
                print("Time Taken:", end_time - start_time)
                return moves

            if curr.cost + 1 <= cost_limit:
                child_cost = curr.cost + 1
                for i in range(12):
                    nodes_expanded += 1
                    new_node = Node()
                    new_node.cube = next_state(curr.cube, i + 1)
                    new_node.cost = child_cost
                    new_node.location = next_location(curr.location, i + 1)
                    new_node.heuristic = heuristic(new_node.location)
                    new_node.parent = curr
                    new_node.move = i + 1
                    heapq.heappush(
                        frontier, (new_node.cost + new_node.heuristic, new_node)
                    )

            cost_limit += 1

    elif method == "BiBFS":
        cost_limit = 1
        nodes_explored = 0
        nodes_expanded = 0
        depth_of_answer = 0
        moves = list()
        frontier_start = OrderedDict()
        frontier_goal = OrderedDict()
        solved = solved_state()

        start = Node()
        start.cube = init_state
        frontier_start[hash(start.cube.tobytes())] = start
        goal = Node()
        goal.cube = solved
        frontier_goal[hash(goal.cube.tobytes())] = goal

        start_time = time.time()

        while len(frontier_start) != 0 and len(frontier_goal) != 0:
            # Forward search from start
            key_start, curr_start = frontier_start.popitem(last=False)
            nodes_explored += 1

            if hash(curr_start.cube.tobytes()) in frontier_goal:
                # The two searches meet
                curr_goal = frontier_goal[hash(curr_start.cube.tobytes())]
                print("Meet in the middle")
                # Retrieve moves from the initial state to the meeting point
                backing = frontier_goal[hash(curr_start.cube.tobytes())]
                front = curr_start
                moves_start = []

                while front.parent is not None:
                    moves_start.append(front.move)
                    front = front.parent

                moves_start.reverse()

                # Retrieve moves from the meeting point to the goal state
                moves_goal = []
                while backing.parent is not None:
                    if backing.move > 6:
                        moves_goal.append(backing.move - 6)
                    else:
                        moves_goal.append(backing.move + 6)
                    backing = backing.parent

                moves = moves_start + moves_goal

                depth_of_answer = len(moves)  # Calculate the depth of the answer

                print("Nodes Explored:", nodes_explored)
                print("Nodes Expanded:", nodes_expanded)
                print("Depth of Answer:", depth_of_answer)
                print("Time Taken:", time.time() - start_time)

                return moves

            if curr_start.cost + 1 <= cost_limit:
                child_cost = curr_start.cost + 1
                for i in range(12):
                    nodes_expanded += 1
                    new_start = Node()
                    new_start.cube = next_state(curr_start.cube, i + 1)
                    new_start.cost = child_cost
                    new_start.parent = curr_start
                    new_start.move = i + 1
                    frontier_start[hash(new_start.cube.tobytes())] = new_start

            # Backward search from goal
            key_goal, curr_goal = frontier_goal.popitem(last=False)
            nodes_explored += 1

            if hash(curr_goal.cube.tobytes()) in frontier_start:
                # The two searches meet
                curr_start = frontier_start[hash(curr_goal.cube.tobytes())]
                print("Meet in the middle_2")
                # Further processing if needed
                front = frontier_start[hash(curr_goal.cube.tobytes())]
                backing = curr_goal
                moves_start = []

                while front.parent is not None:
                    moves_start.append(front.move)
                    front = front.parent

                moves_start.reverse()

                moves_goal = []
                while backing.parent is not None:
                    if backing.move > 6:
                        moves_goal.append(backing.move - 6)
                    else:
                        moves_goal.append(backing.move + 6)
                    backing = backing.parent

                moves = moves_start + moves_goal

                depth_of_answer = len(moves)  # Calculate the depth of the answer

                print("Nodes Explored:", nodes_explored)
                print("Nodes Expanded:", nodes_expanded)
                print("Depth of Answer:", depth_of_answer)
                print("Time Taken:", time.time() - start_time)

                return moves

            if curr_goal.cost + 1 <= cost_limit:
                child_cost = curr_goal.cost + 1
                for i in range(12):
                    nodes_expanded += 1
                    new_goal = Node()
                    new_goal.cube = next_state(curr_goal.cube, i + 1)
                    new_goal.cost = child_cost
                    new_goal.parent = curr_goal
                    new_goal.move = i + 1
                    frontier_goal[hash(new_goal.cube.tobytes())] = new_goal

            cost_limit += 1

        return 0

    else:
        return []
