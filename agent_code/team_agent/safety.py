"""Filter actions with no survival path under the observed bomb schedule.

This is a conservative safety constraint, not a target-seeking policy. Learned
Q-values select among the remaining actions. Crates and opponents are held
fixed while planning; future opponent movement and bomb drops are unknown.
Blast rays match this framework: walls stop explosions, crates do not.
"""
import numpy as np
import settings as s
from .features import DELTA, _bomb_blast_coords, legal_action_mask


def _survival_tiles(state, place_bomb=False, bomb_margin=0, buffered_bomb_position=None):
    field = state['field']
    start = state['self'][3]
    bombs = list(state['bombs'])
    if place_bomb:
        bombs.append((start, max(0, s.BOMB_TIMER - bomb_margin)))
    # A timer-zero bomb detonates after the actions in the next step. A newly
    # placed timer-four bomb is decremented in its placement step, so it
    # detonates at relative step five, after four opportunities to move.
    schedules = [(pos, max(0, int(timer)) + 1) for pos, timer in bombs]
    horizon = max([int(np.max(state['explosion_map'])), 1] +
                  [detonation + s.EXPLOSION_TIMER for _, detonation in schedules])
    hazards = [np.zeros(field.shape, dtype=bool) for _ in range(horizon + 1)]
    blocked = [np.zeros(field.shape, dtype=bool) for _ in range(horizon + 1)]
    for t in range(1, horizon + 1):
        hazards[t] |= state['explosion_map'] >= t
    for pos, detonation in schedules:
        for t in range(1, min(horizon + 1, detonation + 1)):
            blocked[t][pos] = True
        blast = _bomb_blast_coords(field, *pos)
        # For the optional ongoing buffer, retain the real hazard window and
        # additionally start the tracked own bomb's hazard one step earlier.
        # Occupancy and the actual game's countdown remain unchanged.
        hazard_start = max(1, detonation - 1) if pos == buffered_bomb_position else detonation
        for t in range(hazard_start, min(horizon + 1, detonation + s.EXPLOSION_TIMER)):
            for tile in blast:
                hazards[t][tile] = True
    free = field == 0
    for other in state['others']:
        free[other[3]] = False
    # Backward dynamic programming over (time, tile). Staying on a bomb is
    # allowed, but moving into a bomb tile is blocked until detonation.
    viable = free & ~hazards[horizon]
    for t in range(horizon - 1, 0, -1):
        destinations = viable & ~blocked[t + 1]
        reachable = viable.copy()  # WAIT
        reachable[:-1, :] |= destinations[1:, :]
        reachable[1:, :] |= destinations[:-1, :]
        reachable[:, :-1] |= destinations[:, 1:]
        reachable[:, 1:] |= destinations[:, :-1]
        viable = reachable & free & ~hazards[t]
    return viable, blocked[1]


def _survival_path(state, action, bomb_margin=0):
    viable, blocked = _survival_tiles(state, action == 'BOMB', bomb_margin)
    start = state['self'][3]
    dx, dy = DELTA.get(action, (0, 0))
    first = (start[0] + dx, start[1] + dy)
    if not (0 <= first[0] < viable.shape[0] and 0 <= first[1] < viable.shape[1]):
        return False
    return bool(viable[first] and (first == start or not blocked[first]))


def survival_action_mask(state, actions, bomb_margin=0):
    legal = legal_action_mask(state, actions)
    safe = legal.copy()
    threatened = bool(state['bombs']) or bool(np.any(state['explosion_map'] > 0))
    if threatened:
        viable, blocked = _survival_tiles(state)
        start = state['self'][3]
    for i, action in enumerate(actions):
        if legal[i] and action == 'BOMB':
            # An optional earlier hypothetical detonation requires spare escape
            # time before dropping a bomb. Existing bombs keep their real timers.
            safe[i] = _survival_path(state, action, bomb_margin)
        elif legal[i] and threatened:
            dx, dy = DELTA.get(action, (0, 0))
            first = (start[0] + dx, start[1] + dy)
            safe[i] = viable[first] and (first == start or not blocked[first])
    # If static assumptions predict unavoidable death, still choose a learned
    # legal action. Opponents might move, making the prediction pessimistic.
    return safe if safe.any() else legal


def buffered_survival_action_mask(state, actions, own_bomb_position, bomb_margin=1):
    """Keep one step of escape slack for an observed, tracked own bomb.

    If the stronger constraint has no viable action, retain the historical
    mask rather than introducing a new emergency fallback in this experiment.
    """
    ordinary = survival_action_mask(state, actions, bomb_margin)
    if own_bomb_position is None or not any(pos == own_bomb_position for pos, _ in state['bombs']):
        return ordinary
    viable, blocked = _survival_tiles(state, buffered_bomb_position=own_bomb_position)
    start = state['self'][3]
    buffered = ordinary.copy()
    for i, action in enumerate(actions):
        if buffered[i]:
            dx, dy = DELTA.get(action, (0, 0))
            first = (start[0] + dx, start[1] + dy)
            buffered[i] = viable[first] and (first == start or not blocked[first])
    return buffered if buffered.any() else ordinary
