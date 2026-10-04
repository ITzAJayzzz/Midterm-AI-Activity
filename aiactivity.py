"""
Search Module: BFS, DFS and A* Search
-------------------------------------
Graph is based on the whiteboard sketch (edit GRAPH / HEURISTIC to match yours).
Start = S, Goal = G
"""

import heapq
from collections import deque

# ---------------------------------------------------------------
# 1. GRAPH (undirected, weighted) -> adjacency list
#    node: [(neighbor, edge_cost), ...]
# ---------------------------------------------------------------
EDGES = [
    ("S", "A", 3),
    ("A", "G", 1),
    ("S", "B", 2),
    ("B", "G", 8),
    ("S", "D", 4),
    ("D", "C", 5),
    ("C", "G", 6),
]

GRAPH = {}
for u, v, w in EDGES:
    GRAPH.setdefault(u, []).append((v, w))
    GRAPH.setdefault(v, []).append((u, w))

# h(n): estimated cost from each node to the goal (must never overestimate)
HEURISTIC = {"S": 4, "A": 1, "B": 6, "C": 5, "D": 7, "G": 0}

START, GOAL = "S", "G"


def path_cost(path):
    """Sum of edge costs along a path.

    Returns 0 for empty or one-node paths and raises a clear ValueError
    if the path includes an invalid edge transition.
    """
    if not path or len(path) < 2:
        return 0

    total = 0
    for a, b in zip(path, path[1:]):
        neighbors = GRAPH.get(a, [])
        edge_cost = next((cost for neighbor, cost in neighbors if neighbor == b), None)
        if edge_cost is None:
            raise ValueError(f"Invalid path segment: {a} -> {b} is not an edge in the graph.")
        total += edge_cost
    return total


# ---------------------------------------------------------------
# 2. BREADTH-FIRST SEARCH  (queue, FIFO)
#    Finds the path with the FEWEST edges (ignores weights).
# ---------------------------------------------------------------
def bfs(start, goal):
    frontier = deque([[start]])      # queue of paths
    visited = {start}
    order = []                       # order nodes are expanded

    while frontier:
        path = frontier.popleft()    # take from the FRONT
        node = path[-1]
        order.append(node)

        if node == goal:
            return path, order

        for neighbor, _ in GRAPH[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(path + [neighbor])   # add to the BACK
    return None, order


# ---------------------------------------------------------------
# 3. DEPTH-FIRST SEARCH  (stack, LIFO)
#    Goes as deep as possible first. Not guaranteed shortest.
# ---------------------------------------------------------------
def dfs(start, goal):
    frontier = [[start]]             # stack of paths
    visited = set()
    order = []

    while frontier:
        path = frontier.pop()        # take from the TOP
        node = path[-1]

        if node in visited:
            continue
        visited.add(node)
        order.append(node)

        if node == goal:
            return path, order

        # reversed() so neighbors are explored in the order they are listed
        for neighbor, _ in reversed(GRAPH[node]):
            if neighbor not in visited:
                frontier.append(path + [neighbor])
    return None, order


# ---------------------------------------------------------------
# 4. A* SEARCH  (priority queue ordered by f(n) = g(n) + h(n))
#    g(n) = actual cost from start to n
#    h(n) = estimated cost from n to goal
#    Finds the CHEAPEST path when h(n) is admissible.
# ---------------------------------------------------------------
def a_star(start, goal, verbose=True):
    # heap items: (f, g, node, path)
    frontier = [(HEURISTIC[start], 0, start, [start])]
    best_g = {start: 0}              # cheapest g found so far per node
    order = []

    while frontier:
        f, g, node, path = heapq.heappop(frontier)   # lowest f first

        if g > best_g.get(node, float("inf")):
            continue                 # outdated entry, skip

        order.append(node)
        if verbose:
            print(f"  expand {node}: g={g}, h={HEURISTIC[node]}, f={f}")

        if node == goal:
            return path, order, g

        for neighbor, cost in GRAPH[node]:
            new_g = g + cost
            if new_g < best_g.get(neighbor, float("inf")):
                best_g[neighbor] = new_g
                new_f = new_g + HEURISTIC[neighbor]
                heapq.heappush(frontier, (new_f, new_g, neighbor, path + [neighbor]))
    return None, order, float("inf")


# ---------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------
if __name__ == "__main__":
    print(f"Start = {START}, Goal = {GOAL}\n")

    path, order = bfs(START, GOAL)
    print("BFS")
    print("  expansion order:", " -> ".join(order))
    print("  path found     :", " -> ".join(path), f"(cost {path_cost(path)})\n")

    path, order = dfs(START, GOAL)
    print("DFS")
    print("  expansion order:", " -> ".join(order))
    print("  path found     :", " -> ".join(path), f"(cost {path_cost(path)})\n")

    print("A*")
    path, order, cost = a_star(START, GOAL)
    print("  expansion order:", " -> ".join(order))
    print("  path found     :", " -> ".join(path), f"(cost {cost})")