from collections import deque

import numpy as np
import settings as s

FEATURE_VERSION = "v4_stage3_opponent_hunt_bomb_escape"

DIRECTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
DELTA = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}


def legal_action_mask(game_state, actions):
    """Exclude actions blocked in the observed state; WAIT always remains legal.

    Opponents can move later in the same step, so this cannot prevent every
    collision. This checks physical legality, not whether an action is good.
    """
    mask = np.ones(len(actions), dtype=bool)
    if game_state is None:
        return mask
    field = game_state['field']
    free = _free_mask(field, game_state)
    x, y = game_state['self'][3]
    for i, action in enumerate(actions):
        if action in DELTA:
            dx, dy = DELTA[action]
            nx, ny = x + dx, y + dy
            mask[i] = (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]
                       and free[nx, ny])
        elif action == 'BOMB':
            mask[i] = bool(game_state['self'][2])
    return mask

N_FEATURES = 36
N_ENHANCED_FEATURES = 57
# 4 free-direction flags + 5-way coin-direction one-hot + coin distance + bias        (11, stage 1)
# 4 danger-direction flags + 1 danger-here flag                                       (5)
# 5-way escape-direction one-hot (nearest safe tile)                                  (5)
# 5-way crate-direction one-hot + crate distance                                      (6)
# 1 bomb-available flag                                                               (1)
# 1 flag: would an escape route still exist if I bombed right now                     (1)
# 5-way opponent-direction one-hot + opponent distance                                (6, stage 3)
# 1 flag: would a bomb dropped here reach at least one opponent                       (1)


def _bfs_direction_to_nearest(free: np.ndarray, start: tuple, targets: list):
    """
    First move on the shortest path from start to the nearest target, using
    only tiles where `free` is True. Returns (direction, distance),
    ('HERE', 0) if already on a target, or (None, None) if nothing is reachable.
    """
    if not targets:
        return None, None

    target_set = set(targets)
    if start in target_set:
        return 'HERE', 0

    width, height = free.shape

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


def _direction_onehot(direction):
    onehot = np.zeros(5, dtype=np.float32)  # UP, DOWN, LEFT, RIGHT, NONE
    dir_index = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3}
    onehot[dir_index.get(direction, 4)] = 1.0
    return onehot


def _bomb_blast_coords(field: np.ndarray, x: int, y: int) -> list:
    coords = [(x, y)]
    for dx, dy in DELTA.values():
        for i in range(1, s.BOMB_POWER + 1):
            nx, ny = x + dx * i, y + dy * i
            if field[nx, ny] == -1:
                break
            coords.append((nx, ny))
    return coords


def _danger_tiles(field: np.ndarray, game_state: dict) -> set:
    danger = set()
    for (bx, by), _timer in game_state['bombs']:
        danger.update(_bomb_blast_coords(field, bx, by))

    explosion_map = game_state['explosion_map']
    danger.update(zip(*np.nonzero(explosion_map > 0)))

    return danger


def _free_mask(field: np.ndarray, game_state: dict, block_opponents: bool = True) -> np.ndarray:
    """Walkable tiles: static field (walls/crates) minus tiles a bomb currently
    occupies (a bomb blocks its tile until it explodes) and, by default, tiles
    an opponent currently occupies (you can't walk through another agent).
    `block_opponents=False` is for the opponent-direction BFS itself, where an
    opponent's own tile must still count as a reachable target."""
    free = field == 0
    for (bx, by), _timer in game_state['bombs']:
        free[bx, by] = False
    if block_opponents:
        for _, _, _, (ox, oy) in game_state['others']:
            free[ox, oy] = False
    return free


def _escape_exists_if_bombed(field: np.ndarray, game_state: dict, x: int, y: int) -> bool:
    """True if some tile stays reachable and safe after a bomb dropped at (x, y) now."""
    hypothetical_danger = _danger_tiles(field, game_state) | set(_bomb_blast_coords(field, x, y))
    free = _free_mask(field, game_state)
    free[x, y] = False  # The hypothetical bomb occupies the current tile.
    safe_tiles = [tile for tile in zip(*np.nonzero(free)) if tile not in hypothetical_danger]
    direction, _ = _bfs_direction_to_nearest(free, (x, y), safe_tiles)
    return direction is not None


def state_to_features(game_state: dict) -> np.ndarray:
    """
    36 floats:
      [0:4]   free-tile flag for UP, DOWN, LEFT, RIGHT
      [4:9]   one-hot direction to nearest reachable coin (UP, DOWN, LEFT, RIGHT, NONE)
      [9]     distance to that coin, normalized to [0, 1] (1.0 = unreachable/none)
      [10]    bias term, always 1.0
      [11:15] danger flag for UP, DOWN, LEFT, RIGHT (bomb blast or live explosion)
      [15]    danger flag for the current tile
      [16:21] one-hot direction to nearest safe tile (UP, DOWN, LEFT, RIGHT, NONE)
      [21:26] one-hot direction to nearest crate-adjacent tile (UP, DOWN, LEFT, RIGHT, NONE)
      [26]    distance to that tile, normalized to [0, 1] (1.0 = unreachable/none)
      [27]    bomb-available flag
      [28]    flag: would an escape route still exist if I bombed the current tile now
      [29:34] one-hot direction to nearest opponent (UP, DOWN, LEFT, RIGHT, NONE)
      [34]    distance to that opponent, normalized to [0, 1] (1.0 = unreachable/none)
      [35]    flag: would a bomb dropped on the current tile reach at least one opponent
    """
    if game_state is None:
        return np.zeros(N_FEATURES, dtype=np.float32)

    field = game_state['field']
    _, _, bombs_left, (x, y) = game_state['self']
    coins = game_state['coins']
    max_dist = float(field.shape[0] + field.shape[1])
    free = _free_mask(field, game_state)

    # --- stage 1: walls, coin direction ---
    free_dirs = np.zeros(4, dtype=np.float32)
    for i, action in enumerate(DIRECTIONS):
        dx, dy = DELTA[action]
        nx, ny = x + dx, y + dy
        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
            free_dirs[i] = 1.0 if free[nx, ny] else 0.0

    coin_direction, coin_distance = _bfs_direction_to_nearest(free, (x, y), coins)
    coin_onehot = _direction_onehot(coin_direction)
    coin_dist_norm = coin_distance / max_dist if coin_distance is not None else 1.0

    # --- stage 2: danger, escape, crates, bombs ---
    danger_tiles = _danger_tiles(field, game_state)

    danger_dirs = np.zeros(4, dtype=np.float32)
    for i, action in enumerate(DIRECTIONS):
        dx, dy = DELTA[action]
        danger_dirs[i] = 1.0 if (x + dx, y + dy) in danger_tiles else 0.0
    danger_here = np.array([1.0 if (x, y) in danger_tiles else 0.0], dtype=np.float32)

    safe_tiles = [tile for tile in zip(*np.nonzero(free)) if tile not in danger_tiles]
    # Escape paths may cross blast tiles before reaching a safe destination.
    escape_direction, _ = _bfs_direction_to_nearest(free, (x, y), safe_tiles)
    escape_onehot = _direction_onehot(escape_direction)

    crates = list(zip(*np.nonzero(field == 1)))
    crate_targets = [
        (cx + dx, cy + dy)
        for (cx, cy) in crates
        for dx, dy in DELTA.values()
        if free[cx + dx, cy + dy]
    ]
    crate_direction, crate_distance = _bfs_direction_to_nearest(free, (x, y), crate_targets)
    crate_onehot = _direction_onehot(crate_direction)
    crate_dist_norm = crate_distance / max_dist if crate_distance is not None else 1.0

    bomb_available = np.array([1.0 if bombs_left else 0.0], dtype=np.float32)

    escape_after_bomb = np.array(
        [1.0 if _escape_exists_if_bombed(field, game_state, x, y) else 0.0],
        dtype=np.float32,
    )

    # --- stage 3: opponents ---
    # own tile must stay reachable as a BFS target, so don't block opponents here
    free_to_opponents = _free_mask(field, game_state, block_opponents=False)
    opponent_positions = [pos for _, _, _, pos in game_state['others']]
    opponent_direction, opponent_distance = _bfs_direction_to_nearest(free_to_opponents, (x, y), opponent_positions)
    opponent_onehot = _direction_onehot(opponent_direction)
    opponent_dist_norm = opponent_distance / max_dist if opponent_distance is not None else 1.0

    blast_here = set(_bomb_blast_coords(field, x, y))
    opponent_in_blast = np.array(
        [1.0 if any(pos in blast_here for pos in opponent_positions) else 0.0],
        dtype=np.float32,
    )

    return np.concatenate([
        free_dirs, coin_onehot, [coin_dist_norm], [1.0],
        danger_dirs, danger_here, escape_onehot,
        crate_onehot, [crate_dist_norm],
        bomb_available,
        escape_after_bomb,
        opponent_onehot, [opponent_dist_norm],
        opponent_in_blast,
    ]).astype(np.float32)


def enhanced_features(game_state, history, base=None):
    """Append history and interactions while preserving all 36 old features.

    Counts describe revisiting neighboring positions, not a prescribed action.
    Interaction features let a linear model learn different values in danger
    and safety. All new weights start at zero when continuing an old model.
    """
    if game_state is None:
        return np.zeros(N_ENHANCED_FEATURES, dtype=np.float32)
    if base is None:
        base = state_to_features(game_state)
    x, y = game_state['self'][3]
    safe = 1.0 - base[15]
    counts = np.array([
        list(history).count((x + DELTA[a][0], y + DELTA[a][1])) / 8.0
        for a in DIRECTIONS
    ], dtype=np.float32)
    counts *= safe * (1.0 - base[11:15])
    # Coin/crate pursuit when safe, escape when threatened, and the conjunction
    # of available bomb + escape route + useful crate/opponent placement.
    crate_here = float(base[26] == 0)
    bomb_ready_safe = base[27] * base[28] * safe
    return np.concatenate([base, counts, base[4:9] * safe,
                           base[21:26] * safe, base[16:21] * base[15],
                           [bomb_ready_safe * crate_here,
                            bomb_ready_safe * base[35]]]).astype(np.float32)
