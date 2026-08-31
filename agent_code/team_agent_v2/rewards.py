import events as e


# ------------------------------------------------------------------
# Custom events
# ------------------------------------------------------------------

MOVED_TOWARD_COIN = "MOVED_TOWARD_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"


# ------------------------------------------------------------------
# Task 1 reward configurations
# ------------------------------------------------------------------

REWARD_CONFIGS = {
    # Sparse baseline:
    # mainly reward actual task success
    "A": {
        e.COIN_COLLECTED: 10,
        e.KILLED_SELF: -20,
    },

    # Add penalties for behaviour that is unnecessary in Task 1
    "B": {
        e.COIN_COLLECTED: 10,
        e.INVALID_ACTION: -1,
        e.WAITED: -0.5,
        e.BOMB_DROPPED: -2,
        e.KILLED_SELF: -20,
    },

    # Add directional reward shaping
    "C": {
        e.COIN_COLLECTED: 10,
        e.INVALID_ACTION: -1,
        e.WAITED: -0.5,
        e.BOMB_DROPPED: -2,
        e.KILLED_SELF: -20,
        MOVED_TOWARD_COIN: 0.2,
        MOVED_AWAY_FROM_COIN: -0.2,
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


def add_custom_events(
    old_game_state,
    self_action,
    new_game_state,
    events,
):
    """
    Add custom reward events based on the transition between two states.

    Currently used for Task 1 to determine whether the agent moved
    toward or away from the nearest visible coin.

    A new list is returned so that the original framework event list
    is not modified unexpectedly.
    """
    all_events = list(events)

    if old_game_state is None or new_game_state is None:
        return all_events

    # Collecting the coin already receives the stronger outcome reward.
    if e.COIN_COLLECTED in all_events:
        return all_events

    movement_actions = ["UP", "RIGHT", "DOWN", "LEFT"]

    if self_action not in movement_actions:
        return all_events

    coins = old_game_state["coins"]

    if not coins:
        return all_events

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

    return all_events


def reward_from_events(events, config_name="C"):
    """
    Convert a list of game events into one numerical reward.

    config_name:
        A = sparse baseline
        B = behaviour penalties
        C = directional reward shaping
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
