import numpy as np

import events as e
import settings as s
from .features import _danger_tiles, _escape_exists_if_bombed


# ------------------------------------------------------------------
# Custom events
# ------------------------------------------------------------------

MOVED_TOWARD_COIN = "MOVED_TOWARD_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"
USEFUL_BOMB_DROPPED = "USEFUL_BOMB_DROPPED"
USELESS_BOMB_DROPPED = "USELESS_BOMB_DROPPED"
ESCAPED_DANGER = "ESCAPED_DANGER"
ENTERED_DANGER = "ENTERED_DANGER"
STAYED_IN_DANGER = "STAYED_IN_DANGER"
MOVED_TOWARD_SAFETY = "MOVED_TOWARD_SAFETY"
MOVED_AWAY_FROM_SAFETY = "MOVED_AWAY_FROM_SAFETY"
BOMB_WITHOUT_ESCAPE = "BOMB_WITHOUT_ESCAPE"


# ------------------------------------------------------------------
# Task 1 reward configurations
# ------------------------------------------------------------------

REWARD_CONFIGS = {
    "S2_A": {
        e.COIN_COLLECTED: 10,
        e.CRATE_DESTROYED: 2,
        e.COIN_FOUND: 2,
        e.KILLED_SELF: -20,
    },

    "S2_B": {
        e.COIN_COLLECTED: 10,
        e.CRATE_DESTROYED: 2,
        e.COIN_FOUND: 2,
        e.KILLED_SELF: -20,
        e.INVALID_ACTION: -1,
        e.WAITED: -0.5,
        USEFUL_BOMB_DROPPED: 0.5,
        USELESS_BOMB_DROPPED: -0.5,
    },

    "S2_C": {
        e.COIN_COLLECTED: 10,
        e.CRATE_DESTROYED: 2,
        e.COIN_FOUND: 2,
        e.KILLED_SELF: -20,
        e.INVALID_ACTION: -1,
        e.WAITED: -0.5,
        USEFUL_BOMB_DROPPED: 0.5,
        USELESS_BOMB_DROPPED: -0.5,
        ESCAPED_DANGER: 1.0,
        ENTERED_DANGER: -1.0,
        STAYED_IN_DANGER: -0.2,
    },

    # S2_C + graduated feedback while still in danger (were flat -0.2
    # regardless of direction) - local experiment, not Aleksandra's config
    "S2_D": {
        e.COIN_COLLECTED: 10,
        e.CRATE_DESTROYED: 2,
        e.COIN_FOUND: 2,
        e.KILLED_SELF: -20,
        e.INVALID_ACTION: -1,
        e.WAITED: -0.5,
        USEFUL_BOMB_DROPPED: 0.5,
        USELESS_BOMB_DROPPED: -0.5,
        ESCAPED_DANGER: 1.0,
        ENTERED_DANGER: -1.0,
        STAYED_IN_DANGER: -0.2,
        MOVED_TOWARD_SAFETY: 0.3,
        MOVED_AWAY_FROM_SAFETY: -0.3,
    },

    # S2_D + penalize bombing into a spot with no escape route - catches the
    # actual bad decision instead of only punishing the death several steps later
    "S2_E": {
        e.COIN_COLLECTED: 10,
        e.CRATE_DESTROYED: 2,
        e.COIN_FOUND: 2,
        e.KILLED_SELF: -20,
        e.INVALID_ACTION: -1,
        e.WAITED: -0.5,
        USEFUL_BOMB_DROPPED: 0.5,
        USELESS_BOMB_DROPPED: -0.5,
        ESCAPED_DANGER: 1.0,
        ENTERED_DANGER: -1.0,
        STAYED_IN_DANGER: -0.2,
        MOVED_TOWARD_SAFETY: 0.3,
        MOVED_AWAY_FROM_SAFETY: -0.3,
        BOMB_WITHOUT_ESCAPE: -15,
    },
}


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def manhattan_distance(position1, position2):
    """Return Manhattan distance between two board positions."""
    return (
        abs(position1[0] - position2[0])
        + abs(position1[1] - position2[1])
    )

def bomb_would_hit_crate(game_state):
    """
    Return True if a bomb placed at the agent's current position
    would hit at least one crate.
    """
    if game_state is None:
        return False

    field = game_state["field"]
    x, y = game_state["self"][3]

    directions = [
        (0, -1),
        (0, 1),
        (-1, 0),
        (1, 0),
    ]

    bomb_power = s.BOMB_POWER

    for dx, dy in directions:
        for distance in range(1, bomb_power + 1):
            nx = x + dx * distance
            ny = y + dy * distance

            tile = field[nx, ny]

            # Solid wall blocks the blast.
            if tile == -1:
                break

            # A crate would be destroyed and also blocks the blast.
            if tile == 1:
                return True

    return False


def add_custom_events(
    old_game_state,
    self_action,
    new_game_state,
    events,
):
    """
    Add custom reward events based on the transition between two states.

    Task 1:
    - movement toward / away from visible coins

    Task 2:
    - useful / useless bomb placement
    - escaping from bomb danger
    - entering or remaining in danger

    A new list is returned so that the original framework event list
    is not modified unexpectedly.
    """
    all_events = list(events)

    if old_game_state is None or new_game_state is None:
        return all_events

    movement_actions = ["UP", "RIGHT", "DOWN", "LEFT"]

    # --------------------------------------------------------------
    # Task 1: movement toward visible coins
    # --------------------------------------------------------------

    if (
        self_action in movement_actions
        and e.COIN_COLLECTED not in all_events
    ):
        coins = old_game_state["coins"]

        if coins:
            old_position = old_game_state["self"][3]
            new_position = new_game_state["self"][3]

            nearest_coin = min(
                coins,
                key=lambda coin: manhattan_distance(old_position, coin),
            )

            old_distance = manhattan_distance(
                old_position,
                nearest_coin,
            )

            new_distance = manhattan_distance(
                new_position,
                nearest_coin,
            )

            if new_distance < old_distance:
                all_events.append(MOVED_TOWARD_COIN)

            elif new_distance > old_distance:
                all_events.append(MOVED_AWAY_FROM_COIN)

    # --------------------------------------------------------------
    # Task 2: bomb placement
    # --------------------------------------------------------------

    if (
        self_action == "BOMB"
        and e.BOMB_DROPPED in all_events
    ):
        if bomb_would_hit_crate(old_game_state):
            all_events.append(USEFUL_BOMB_DROPPED)
        else:
            all_events.append(USELESS_BOMB_DROPPED)

        field = old_game_state["field"]
        x, y = old_game_state["self"][3]
        if not _escape_exists_if_bombed(field, old_game_state, x, y):
            all_events.append(BOMB_WITHOUT_ESCAPE)

    # --------------------------------------------------------------
    # Task 2: danger / escape behaviour
    # --------------------------------------------------------------

    if self_action in movement_actions:
        old_danger = is_in_danger(old_game_state)
        new_danger = is_in_danger(new_game_state)

        if old_danger and not new_danger:
            all_events.append(ESCAPED_DANGER)

        elif not old_danger and new_danger:
            all_events.append(ENTERED_DANGER)

        elif old_danger and new_danger:
            all_events.append(STAYED_IN_DANGER)

            old_dist = nearest_safe_tile_distance(old_game_state)
            new_dist = nearest_safe_tile_distance(new_game_state)
            if old_dist is not None and new_dist is not None:
                if new_dist < old_dist:
                    all_events.append(MOVED_TOWARD_SAFETY)
                elif new_dist > old_dist:
                    all_events.append(MOVED_AWAY_FROM_SAFETY)

    return all_events

def nearest_safe_tile_distance(game_state):
    """Manhattan distance from the agent to the nearest non-dangerous tile."""
    field = game_state["field"]
    danger = _danger_tiles(field, game_state)
    free = field == 0
    safe_tiles = [tile for tile in zip(*np.nonzero(free)) if tile not in danger]

    if not safe_tiles:
        return None

    position = game_state["self"][3]
    return min(manhattan_distance(position, tile) for tile in safe_tiles)


def is_in_danger(game_state):
    """Return True if the agent is currently on a dangerous tile."""
    if game_state is None:
        return False

    field = game_state["field"]
    position = game_state["self"][3]

    return position in _danger_tiles(field, game_state)


def reward_from_events(events, config_name="C"):
    """
    Convert a list of game events into one numerical reward.

    config_name:
        A = sparse baseline
        B = behaviour penalties
        C = directional reward shaping
        
        S2_A = Task 2 outcome rewards
        S2_B = Task 2 bomb-placement shaping
        S2_C = Task 2 bomb-placement + escape shaping
    """
    if config_name not in REWARD_CONFIGS:
        raise ValueError(
            f"Unknown reward configuration: {config_name}. "
            f"Choose one of {list(REWARD_CONFIGS)}."
        )

    reward_mapping = REWARD_CONFIGS[config_name]

    reward_sum = 0.0

    for event in events:
        reward_sum += reward_mapping.get(event, 0)

    return reward_sum