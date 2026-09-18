"""Checks for legality and for excluding impossible actions from TD targets."""
import unittest
from collections import deque
from types import SimpleNamespace
import numpy as np
from agent_code.team_agent.config import ACTIONS
from agent_code.team_agent.features import legal_action_mask, enhanced_features, state_to_features
from agent_code.team_agent.model import LinearQModel
from agent_code.team_agent.safety import survival_action_mask, _survival_path, buffered_survival_action_mask
from agent_code.team_agent.callbacks import act, policy_features


class ActionMaskTests(unittest.TestCase):
    @staticmethod
    def corridor_state(escape=True):
        field = -np.ones((9, 9), dtype=int)
        field[1:6, 3] = 0
        if not escape:
            field[5, 3] = 1
        return {'field': field, 'self': ('test', 0, True, (1, 3)),
                'others': [], 'bombs': [], 'coins': [],
                'explosion_map': np.zeros_like(field)}

    def test_new_bomb_can_be_outrun_in_four_moves(self):
        self.assertTrue(_survival_path(self.corridor_state(), 'BOMB'))

    def test_bomb_margin_requires_spare_time(self):
        state = self.corridor_state()
        self.assertTrue(_survival_path(state, 'BOMB', 0))
        self.assertFalse(_survival_path(state, 'BOMB', 1))

    def test_ongoing_buffer_excludes_wait_that_consumes_last_spare_step(self):
        state = self.corridor_state()
        state['self'] = ('test', 0, False, (2, 3))
        state['bombs'] = [((1, 3), 3)]
        self.assertTrue(survival_action_mask(state, ACTIONS, 1)[ACTIONS.index('WAIT')])
        mask = buffered_survival_action_mask(state, ACTIONS, (1, 3), 1)
        self.assertFalse(mask[ACTIONS.index('WAIT')])
        self.assertTrue(mask[ACTIONS.index('RIGHT')])

    def test_buffer_does_not_apply_to_untracked_opponent_bomb(self):
        state = self.corridor_state()
        state['self'] = ('test', 0, False, (2, 3))
        state['bombs'] = [((1, 3), 3)]
        np.testing.assert_array_equal(buffered_survival_action_mask(state, ACTIONS, None), survival_action_mask(state, ACTIONS, 1))

    def test_own_bomb_tracking_does_not_leak_into_next_round(self):
        state = self.corridor_state()
        state['round'] = 1
        model = LinearQModel(6, 36)
        model.weights[:] = 0
        model.weights[ACTIONS.index('BOMB'), 10] = 10
        policy = SimpleNamespace(model=model, train=False, use_legal_action_mask=True,
                                 use_survival_action_mask=False, history_round=None,
                                 position_history=deque(maxlen=8))
        self.assertEqual(act(policy, state), 'BOMB')
        self.assertEqual(policy.own_bomb_position, (1, 3))
        next_state = dict(state, self=('test', 0, False, (2, 3)), bombs=[((1, 3), 3)])
        act(policy, next_state)
        self.assertEqual(policy.own_bomb_position, (1, 3))
        next_state['round'] = 2
        act(policy, next_state)
        self.assertIsNone(policy.own_bomb_position)

    def test_impossible_buffer_retains_original_emergency_fallback(self):
        state = self.corridor_state()
        state['self'] = ('test', 0, False, (1, 3))
        state['bombs'] = [((1, 3), 3)]
        original = survival_action_mask(state, ACTIONS, 1)
        np.testing.assert_array_equal(buffered_survival_action_mask(state, ACTIONS, (1, 3)), original)

    def test_bomb_in_sealed_blast_corridor_is_excluded(self):
        state = self.corridor_state(escape=False)
        mask = survival_action_mask(state, ACTIONS)
        self.assertFalse(mask[ACTIONS.index('BOMB')])
        self.assertTrue(mask[ACTIONS.index('WAIT')])

    def test_timer_zero_detonates_after_next_move(self):
        state = self.corridor_state()
        state['self'] = ('test', 0, False, (4, 3))
        state['bombs'] = [((1, 3), 0)]
        mask = survival_action_mask(state, ACTIONS)
        self.assertTrue(mask[ACTIONS.index('RIGHT')])
        self.assertFalse(mask[ACTIONS.index('WAIT')])
        self.assertFalse(mask[ACTIONS.index('LEFT')])

    def test_obstacles_and_bomb_availability(self):
        field = np.zeros((7, 7), dtype=int)
        field[3, 2] = -1
        field[4, 3] = 1
        state = {'field': field, 'self': ('test', 0, False, (3, 3)),
                 'others': [('other', 0, True, (2, 3))], 'bombs': [((3, 4), 2)]}
        mask = legal_action_mask(state, ACTIONS)
        self.assertEqual([a for a, allowed in zip(ACTIONS, mask) if allowed], ['WAIT'])

    def test_masked_td_target_ignores_impossible_maximum(self):
        model = LinearQModel(3, 1, learning_rate=1.0, discount_factor=0.5)
        model.weights[:] = [[0], [100], [4]]
        model.update(np.ones(1), 0, 1, np.ones(1), False, [True, False, True])
        self.assertEqual(model.weights[0, 0], 3.0)
        self.assertEqual(model.best_action(np.ones(1), [True, False, True]), 2)

    def test_terminal_update_does_not_bootstrap(self):
        model = LinearQModel(2, 1, learning_rate=1.0)
        model.weights[:] = 100
        model.update(np.ones(1), 0, -7, None, True)
        self.assertEqual(model.weights[0, 0], -7)

    def test_expansion_preserves_existing_policy_values(self):
        field = np.zeros((7, 7), dtype=int)
        field[0, :] = field[-1, :] = -1
        field[:, 0] = field[:, -1] = -1
        state = {'field': field, 'self': ('test', 0, True, (3, 3)),
                 'others': [], 'bombs': [], 'coins': [(1, 1)],
                 'explosion_map': np.zeros_like(field)}
        base = state_to_features(state)
        extended = enhanced_features(state, [(3, 2), (3, 4)])
        self.assertEqual(extended.shape, (57,))
        model = LinearQModel(6, 36, seed=19)
        before = model.q_values(base)
        model.expand_features(57)
        np.testing.assert_allclose(model.q_values(extended), before, atol=1e-7)

    def test_history_used_by_next_td_target_matches_next_action(self):
        state = self.corridor_state()
        state['round'] = 1
        model = LinearQModel(6, 57, seed=1)
        policy = SimpleNamespace(model=model, train=False, use_legal_action_mask=True,
                                 use_survival_action_mask=False, history_round=None,
                                 position_history=deque(maxlen=8))
        act(policy, state)
        next_state = dict(state, self=('test', 0, True, (2, 3)))
        td_features = policy_features(policy, next_state)
        act(policy, next_state)
        np.testing.assert_array_equal(td_features, policy.action_features)
        # A new episode must not inherit the last episode's movement history.
        next_state['round'] = 2
        act(policy, next_state)
        self.assertEqual(tuple(policy.position_history), ((2, 3),))

    def test_safety_countdown_matches_actual_framework(self):
        import logging
        from environment import BombeRLeWorld
        from items import Bomb
        world = BombeRLeWorld.__new__(BombeRLeWorld)
        world.logger = logging.getLogger('safety_test')
        world.arena = self.corridor_state()['field']
        world.coins = []
        world.explosions = []
        owner = SimpleNamespace(name='test', bombs_left=False, add_event=lambda event: None)
        world.bombs = [Bomb((1, 3), owner, 4, 3, None)]
        for _ in range(4):
            world.update_bombs()
            self.assertFalse(world.explosions)
        world.update_bombs()
        self.assertTrue(world.explosions[0].is_dangerous())
        world.update_explosions()
        self.assertTrue(world.explosions[0].is_dangerous())
        world.update_explosions()
        self.assertFalse(world.explosions[0].is_dangerous())


if __name__ == '__main__':
    unittest.main()
