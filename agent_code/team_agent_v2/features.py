from collections import deque

import numpy as np

FEATURE_VERSION = "v1_stage1_nav_coin"

DIRECTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
DELTA = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}

N_FEATURES = 11  # 4 free-direction flags + 5-way coin-direction one-hot + distance + bias


def _bfs_direction_to_nearest(field: np.ndarray, start: tuple, targets: list):
    """
    First move on the shortest path from start to the nearest target.
    Returns (direction, distance), ('HERE', 0) if already on a target,
    or (None, None) if nothing is reachable.
    """
    if not targets:
        return None, None

    target_set = set(targets)
    if start in target_set:
        return 'HERE', 0

    free = field == 0
    width, height = field.shape

    frontier = deque([start])
    first_action = {start: None}
    dist = {start: 0}

    while frontier:
        current = frontier.popleft()
        for action in DIRECTIONS:
            dx, dy = DELTA[action]
            nxt = (current[0] + dx, current[1] + dy)
            if nxt in first_action:
                continue
            if not (0 <= nxt[0] < width and 0 <= nxt[1] < height):
                continue
            if not free[nxt]:
                continue

            first_action[nxt] = first_action[current] if first_action[current] is not None else action
            dist[nxt] = dist[current] + 1

            if nxt in target_set:
                return first_action[nxt], dist[nxt]

            frontier.append(nxt)

    return None, None


def state_to_features(game_state: dict) -> np.ndarray:
    """
    11 floats:
      [0:4] free-tile flag for UP, DOWN, LEFT, RIGHT
      [4:9] one-hot direction to nearest reachable coin (UP, DOWN, LEFT, RIGHT, NONE)
      [9]   distance to that coin, normalized to [0, 1] (1.0 = unreachable/none)
      [10]  bias term, always 1.0
    """
    if game_state is None:
        return np.zeros(N_FEATURES, dtype=np.float32)

    field = game_state['field']
    _, _, _, (x, y) = game_state['self']
    coins = game_state['coins']

    free_dirs = np.zeros(4, dtype=np.float32)
    for i, action in enumerate(DIRECTIONS):
        dx, dy = DELTA[action]
        nx, ny = x + dx, y + dy
        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
            free_dirs[i] = 1.0 if field[nx, ny] == 0 else 0.0

    direction, distance = _bfs_direction_to_nearest(field, (x, y), coins)

    coin_onehot = np.zeros(5, dtype=np.float32)  # UP, DOWN, LEFT, RIGHT, NONE
    dir_index = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3}
    coin_onehot[dir_index.get(direction, 4)] = 1.0

    max_dist = float(field.shape[0] + field.shape[1])
    norm_dist = distance / max_dist if distance is not None else 1.0

    return np.concatenate([free_dirs, coin_onehot, np.array([norm_dist, 1.0], dtype=np.float32)])
