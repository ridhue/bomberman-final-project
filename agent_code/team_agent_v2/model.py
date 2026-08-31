import numpy as np


class DoubleQModel:
    """
    Double Q-learning: two independent linear estimators, Q(s, a) = w_a . phi(s)
    each. Each update picks one at random to update, using the *other* one to
    evaluate the chosen next action -- decouples action selection from
    evaluation, which reduces the overestimation bias plain Q-learning has.
    """

    def __init__(self, n_actions, n_features, learning_rate=0.05, discount_factor=0.95, seed=None):
        rng = np.random.default_rng(seed)
        self.n_actions = n_actions
        self.n_features = n_features
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.rng = rng
        self.weights_a = rng.normal(scale=0.01, size=(n_actions, n_features)).astype(np.float32)
        self.weights_b = rng.normal(scale=0.01, size=(n_actions, n_features)).astype(np.float32)

    def q_values(self, features: np.ndarray) -> np.ndarray:
        return (self.weights_a @ features + self.weights_b @ features) / 2

    def best_action(self, features: np.ndarray) -> int:
        return int(np.argmax(self.q_values(features)))

    def update(self, features: np.ndarray, action_idx: int, reward: float, next_features, done: bool) -> float:
        if self.rng.random() < 0.5:
            primary, other = self.weights_a, self.weights_b
        else:
            primary, other = self.weights_b, self.weights_a

        q_sa = float(primary[action_idx] @ features)
        if done or next_features is None:
            target = reward
        else:
            best_next = int(np.argmax(primary @ next_features))
            target = reward + self.gamma * float(other[best_next] @ next_features)

        td_error = target - q_sa
        primary[action_idx] += self.alpha * td_error * features
        return td_error
