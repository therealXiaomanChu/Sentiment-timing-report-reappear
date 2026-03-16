import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class EnvState:
    features: np.ndarray
    ret: float


class TradingEnv:
    def __init__(
        self,
        df,
        feature_cols: List[str],
        ret_col: str = 'ret',
        max_position: int = 2,
        transaction_cost_bp: float = 1.0,
        risk_penalty: float = 0.0,
        dd_penalty: float = 0.0,
        clip_reward: float = 1.0,
    ):
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.ret_col = ret_col
        self.max_position = max_position
        self.transaction_cost = transaction_cost_bp / 10000.0
        self.risk_penalty = risk_penalty
        self.dd_penalty = dd_penalty
        self.clip_reward = clip_reward

        self.ptr = 0
        self.position = 0
        self.equity = 1.0
        self.max_equity = 1.0

    def reset(self):
        self.ptr = 0
        self.position = 0
        self.equity = 1.0
        self.max_equity = 1.0
        return self._get_state()

    def step(self, action: int):
        # action in {0..2*max_position}; map to [-max_position, +max_position]
        target_position = action - self.max_position
        ret = float(self.df.iloc[self.ptr + 1][self.ret_col]) if self.ptr + 1 < len(self.df) else 0.0

        # transaction cost proportional to position change
        turnover = abs(target_position - self.position)
        cost = turnover * self.transaction_cost

        # update position and equity
        self.position = target_position
        reward = self.position * ret - cost
        self.equity *= (1.0 + reward)
        self.max_equity = max(self.max_equity, self.equity)

        # risk/drawdown penalties
        drawdown = 0.0 if self.max_equity == 0 else 1.0 - (self.equity / self.max_equity)
        reward -= self.risk_penalty * (abs(self.position) * abs(ret))
        reward -= self.dd_penalty * drawdown

        # clip reward
        reward = float(np.clip(reward, -self.clip_reward, self.clip_reward))

        done = self.ptr + 1 >= len(self.df) - 1
        self.ptr += 1

        return self._get_state(), reward, done, {}

    def action_space_n(self) -> int:
        return 2 * self.max_position + 1

    def _get_state(self) -> np.ndarray:
        features = self.df.iloc[self.ptr][self.feature_cols].values.astype(np.float32)
        return features