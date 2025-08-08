import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Dict, Any, List

from dqn_agent import DQNAgent
from rl_env import TradingEnv


def train_dqn_on_dataframe(
    df,
    feature_cols: List[str],
    n_episodes: int = 50,
    max_position: int = 2,
    epsilon_start: float = 0.8,
    epsilon_end: float = 0.02,
    epsilon_decay_episodes: int = 40,
    gamma: float = 0.99,
    lr: float = 1e-4,
    tau: float = 0.005,
    batch_size: int = 128,
    buffer_capacity: int = 500_000,
    transaction_cost_bp: float = 1.0,
    risk_penalty: float = 0.1,
    dd_penalty: float = 0.2,
    validation_split: float = 0.2,
    early_stopping_patience: int = 50,
    verbose: bool = True,
) -> Dict[str, Any]:
    env = TradingEnv(
        df=df,
        feature_cols=feature_cols,
        ret_col='ret',
        max_position=max_position,
        transaction_cost_bp=transaction_cost_bp,
        risk_penalty=risk_penalty,
        dd_penalty=dd_penalty,
        clip_reward=1.0,
    )

    agent = DQNAgent(
        state_dim=len(feature_cols),
        action_dim=env.action_space_n(),
        hidden_dims=(256, 128),
        lr=lr,
        gamma=gamma,
        tau=tau,
        batch_size=batch_size,
        buffer_capacity=buffer_capacity,
        double_dqn=True,
        dropout=0.1,
    )

    device = agent.device
    replay_states = deque(maxlen=buffer_capacity)
    replay_actions = deque(maxlen=buffer_capacity)
    replay_rewards = deque(maxlen=buffer_capacity)
    replay_next_states = deque(maxlen=buffer_capacity)
    replay_dones = deque(maxlen=buffer_capacity)

    def sample_batch(batch_size: int):
        idx = np.random.choice(len(replay_states), size=batch_size, replace=False)
        states = torch.tensor(np.array([replay_states[i] for i in idx]), dtype=torch.float32, device=device)
        actions = torch.tensor(np.array([replay_actions[i] for i in idx]), dtype=torch.int64, device=device).unsqueeze(1)
        rewards = torch.tensor(np.array([replay_rewards[i] for i in idx]), dtype=torch.float32, device=device).unsqueeze(1)
        next_states = torch.tensor(np.array([replay_next_states[i] for i in idx]), dtype=torch.float32, device=device)
        dones = torch.tensor(np.array([replay_dones[i] for i in idx]), dtype=torch.float32, device=device).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    mse_loss = nn.MSELoss()

    episode_rewards: List[float] = []
    equity_history: List[float] = []

    for ep in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        epsilon = max(
            epsilon_end,
            epsilon_start - (epsilon_start - epsilon_end) * (ep / max(1, epsilon_decay_episodes)),
        )

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            if np.random.rand() < epsilon:
                action = np.random.randint(env.action_space_n())
            else:
                action = agent.act(state_tensor)

            next_state, reward, done, _ = env.step(action)

            replay_states.append(state)
            replay_actions.append(action)
            replay_rewards.append(reward)
            replay_next_states.append(next_state)
            replay_dones.append(done)

            state = next_state
            total_reward += reward
            equity_history.append(env.equity)

            if len(replay_states) >= batch_size:
                states, actions, rewards, next_states, dones = sample_batch(batch_size)
                q_values = agent.q(states).gather(1, actions)

                with torch.no_grad():
                    if agent.double_dqn:
                        next_actions = agent.q(next_states).argmax(dim=1, keepdim=True)
                        next_q = agent.q_target(next_states).gather(1, next_actions)
                    else:
                        next_q = agent.q_target(next_states).max(dim=1, keepdim=True)[0]
                    targets = rewards + (1 - dones) * agent.gamma * next_q

                loss = mse_loss(q_values, targets)
                agent.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.q.parameters(), 1.0)
                agent.optimizer.step()
                agent.update_target()

        episode_rewards.append(total_reward)
        if verbose and (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1}/{n_episodes}, reward: {total_reward:.4f}")

    history = {
        "equity": np.array(equity_history),
        "episode_rewards": np.array(episode_rewards),
    }

    return {
        "agent": agent,
        "final_history": history,
    }