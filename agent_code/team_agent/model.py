import numpy as np


class LinearQModel:
    """Q(s, a) = w_a . phi(s), one weight vector per action, updated via TD(0)."""

    def __init__(self, n_actions, n_features, learning_rate=0.05, discount_factor=0.95, seed=None):
        rng = np.random.default_rng(seed)
        self.n_actions = n_actions
        self.n_features = n_features
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.weights = rng.normal(scale=0.01, size=(n_actions, n_features)).astype(np.float32)

    def q_values(self, features: np.ndarray) -> np.ndarray:
        return self.weights @ features

    def expand_features(self, n_features):
        """Keep learned values unchanged when adding zero-weight features."""
        if n_features < self.n_features:
            raise ValueError('Cannot discard trained features when resuming')
        if n_features > self.n_features:
            self.weights = np.pad(self.weights, ((0, 0), (0, n_features - self.n_features)))
            self.n_features = n_features

    def best_action(self, features: np.ndarray, action_mask=None) -> int:
        values = self.q_values(features)
        if action_mask is not None:
            values = np.where(action_mask, values, -np.inf)
        return int(np.argmax(values))

    def update(self, features: np.ndarray, action_idx: int, reward: float, next_features, done: bool, next_action_mask=None) -> float:
        q_sa = float(self.weights[action_idx] @ features)
        if done or next_features is None:
            target = reward
        else:
            next_values = self.q_values(next_features)
            if next_action_mask is not None:
                next_values = np.where(next_action_mask, next_values, -np.inf)
            target = reward + self.gamma * float(np.max(next_values))
        td_error = target - q_sa
        self.weights[action_idx] += self.alpha * td_error * features
        return td_error
