import time

maze = [
    "S.....",
    ".####.",
    "......",
    ".####.",
    "E.....",
]

# find start and end positions
for r in range(len(maze)):
    for c in range(len(maze[0])):
        if maze[r][c] == "S":
            start = (r, c)
        if maze[r][c] == "E":
            end = (r, c)

print("Start:", start, "End:", end)


def get_neighbours(pos):
    r, c = pos
    result = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < len(maze) and 0 <= nc < len(maze[0]):
            if maze[nr][nc] != "#":
                result.append((nr, nc))
    return result


def bfs(start, end):
    queue = [[start]]          # queue of paths
    visited = [start]
    explored = 0

    while len(queue) > 0:
        path = queue.pop(0)    # take from FRONT
        node = path[-1]
        explored = explored + 1

        if node == end:
            return path, explored

        for n in get_neighbours(node):
            if n not in visited:
                visited.append(n)
                queue.append(path + [n])

    return None, explored


def dfs(start, end):
    stack = [[start]]          # stack of paths
    visited = [start]
    explored = 0

    while len(stack) > 0:
        path = stack.pop()     # take from BACK
        node = path[-1]
        explored = explored + 1

        if node == end:
            return path, explored

        for n in get_neighbours(node):
            if n not in visited:
                visited.append(n)
                stack.append(path + [n])

    return None, explored


t1 = time.perf_counter()
bfs_path, bfs_nodes = bfs(start, end)
bfs_time = (time.perf_counter() - t1) * 1000

t2 = time.perf_counter()
dfs_path, dfs_nodes = dfs(start, end)
dfs_time = (time.perf_counter() - t2) * 1000

print("Algorithm | Path length | Nodes explored | Time (ms)")
print("BFS       |", len(bfs_path), "         |", bfs_nodes, "           |", round(bfs_time, 4))
print("DFS       |", len(dfs_path), "         |", dfs_nodes, "           |", round(dfs_time, 4))


def show_path(path):
    grid = [list(row) for row in maze]
    for (r, c) in path:
        if grid[r][c] == ".":
            grid[r][c] = "*"
    for row in grid:
        print("".join(row))

print("BFS path:")
show_path(bfs_path)
print()
print("DFS path:")
show_path(dfs_path)
