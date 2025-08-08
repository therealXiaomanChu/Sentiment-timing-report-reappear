import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

from dqn_agent import DQNAgent
from rl_env import TradingEnv

class DQNSentimentModel:
    def __init__(self, factors_df):
        self.factors_df = factors_df
        self.sentiment_scores = None
        self.feature_cols = [col for col in factors_df.columns if col != '市场收益']
        self.scaler = StandardScaler()
        self.setup_environment()
        
    def setup_environment(self):
        """设置强化学习环境"""
        # 准备数据
        df = self.factors_df.copy()
        df['ret'] = df['市场收益'] / 100  # 转换为小数
        
        # 设置环境参数
        self.env = TradingEnv(
            df=df,
            feature_cols=self.feature_cols,
            ret_col='ret',
            max_position=2,  # 允许的最大仓位范围：-2到+2
            transaction_cost_bp=1.0,  # 1bp交易成本
            risk_penalty=0.1,  # 风险惩罚系数
            dd_penalty=0.2,  # 回撤惩罚系数
            clip_reward=1.0  # 限制奖励范围
        )
        
        # 设置DQN智能体
        self.agent = DQNAgent(
            state_dim=len(self.feature_cols),
            action_dim=self.env.action_space_n(),
            hidden_dims=(512, 256, 128),  # 增加网络深度
            lr=1e-4,
            gamma=0.99,
            tau=0.005,
            batch_size=128,
            buffer_capacity=500_000,
            double_dqn=True,
            dropout=0.2
        )
        
    def create_enhanced_features(self, df):
        """创建增强的特征集"""
        enhanced_df = df.copy()
        
        # 添加时序特征（过去5天）
        for col in self.feature_cols:
            for lag in [1, 2, 3, 5]:
                enhanced_df[f'{col}_lag{lag}'] = df[col].shift(lag)
        
        # 添加滚动统计特征（过去20天）
        for col in self.feature_cols:
            enhanced_df[f'{col}_ma20'] = df[col].rolling(20).mean()
            enhanced_df[f'{col}_std20'] = df[col].rolling(20).std()
            enhanced_df[f'{col}_zscore'] = (df[col] - df[col].rolling(20).mean()) / df[col].rolling(20).std()
        
        # 添加趋势特征
        for col in self.feature_cols:
            enhanced_df[f'{col}_trend5'] = df[col].diff(5)
            enhanced_df[f'{col}_trend10'] = df[col].diff(10)
        
        # 添加交互特征
        enhanced_df['sentiment_momentum'] = enhanced_df['涨停占比'] * enhanced_df['打板收益']
        enhanced_df['fear_greed_ratio'] = enhanced_df['跌停占比'] / (enhanced_df['涨停占比'] + 1e-6)
        enhanced_df['net_sentiment_strength'] = enhanced_df['净涨停占比'] * enhanced_df['打板收益']
        
        # 填充缺失值
        enhanced_df = enhanced_df.fillna(method='bfill').fillna(0)
        
        return enhanced_df
        
    def train_model(self, n_episodes=500, test_size=0.2):
        """训练DQN模型"""
        print("开始训练DQN模型...")
        
        # 创建增强特征
        enhanced_df = self.create_enhanced_features(self.factors_df)
        enhanced_feature_cols = [col for col in enhanced_df.columns if col not in ['ret', '市场收益']]
        
        # 划分训练集和测试集
        train_size = int(len(enhanced_df) * (1 - test_size))
        train_df = enhanced_df.iloc[:train_size].copy()
        test_df = enhanced_df.iloc[train_size:].copy()
        
        # 标准化特征
        train_features = self.scaler.fit_transform(train_df[enhanced_feature_cols])
        test_features = self.scaler.transform(test_df[enhanced_feature_cols])
        
        train_df[enhanced_feature_cols] = train_features
        test_df[enhanced_feature_cols] = test_features
        
        # 更新环境特征
        train_df['ret'] = train_df['市场收益'] / 100
        test_df['ret'] = test_df['市场收益'] / 100
        
        # 训练模型
        train_results = self._train_dqn(train_df, enhanced_feature_cols, n_episodes)
        
        # 生成情绪得分
        self.sentiment_scores = self._generate_improved_sentiment_scores(enhanced_df, enhanced_feature_cols)
        
        # 评估模型性能
        self._evaluate_model_performance(test_df, enhanced_feature_cols)
        
        return train_results
    
    def _train_dqn(self, train_df, feature_cols, n_episodes):
        """DQN训练循环"""
        from train_dqn import train_dqn_on_dataframe
        
        results = train_dqn_on_dataframe(
            df=train_df,
            feature_cols=feature_cols,
            n_episodes=n_episodes,
            max_position=2,
            epsilon_start=0.8,
            epsilon_end=0.02,
            epsilon_decay_episodes=n_episodes * 0.8,
            gamma=0.99,
            lr=1e-4,
            tau=0.005,
            batch_size=128,
            buffer_capacity=500_000,
            transaction_cost_bp=1.0,
            risk_penalty=0.1,
            dd_penalty=0.2,
            validation_split=0.2,
            early_stopping_patience=50,
            verbose=True
        )
        
        # 更新智能体
        self.agent = results['agent']
        
        return results
    
    def _generate_improved_sentiment_scores(self, enhanced_df, feature_cols):
        """改进的情绪得分生成"""
        # 标准化特征
        features = self.scaler.transform(enhanced_df[feature_cols])
        
        with torch.no_grad():
            device = next(self.agent.q.parameters()).device
            state_tensor = torch.tensor(features, dtype=torch.float32, device=device)
            q_values = self.agent.q(state_tensor)
            
            # 使用最优动作的Q值作为情绪得分
            optimal_actions = q_values.argmax(dim=1)
            sentiment_scores = q_values.gather(1, optimal_actions.unsqueeze(1)).squeeze()
            
            # 归一化到[-1, 1]范围
            sentiment_scores = torch.tanh(sentiment_scores)
            
            # 添加置信度权重
            confidence = torch.softmax(q_values, dim=1).max(dim=1)[0]
            sentiment_scores = sentiment_scores * confidence
        
        sentiment_series = pd.Series(sentiment_scores.cpu().numpy(), index=enhanced_df.index)
        
        # 平滑处理
        sentiment_series = sentiment_series.rolling(5, center=True).mean().fillna(method='bfill')
        
        return sentiment_series
    
    def _evaluate_model_performance(self, test_df, feature_cols):
        """评估模型性能"""
        print("\n=== DQN模型性能评估 ===")
        
        # 在测试集上运行模型
        test_features = self.scaler.transform(test_df[feature_cols])
        
        with torch.no_grad():
            device = next(self.agent.q.parameters()).device
            state_tensor = torch.tensor(test_features, dtype=torch.float32, device=device)
            q_values = self.agent.q(state_tensor)
            actions = q_values.argmax(dim=1).cpu().numpy()
        
        # 计算策略收益
        test_returns = test_df['ret'].values[1:]  # 从第二天开始
        strategy_returns = []
        
        for i, action in enumerate(actions[:-1]):  # 最后一天没有下一期收益
            position = action - 2  # 转换为[-2, 2]范围
            strategy_returns.append(position * test_returns[i])
        
        strategy_returns = np.array(strategy_returns)
        
        # 计算性能指标
        total_return = np.prod(1 + strategy_returns) - 1
        annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        volatility = np.std(strategy_returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # 计算最大回撤
        cumulative_returns = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # 计算胜率
        win_rate = np.mean(strategy_returns > 0)
        
        print(f"测试集策略性能:")
        print(f"总收益率: {total_return:.2%}")
        print(f"年化收益率: {annualized_return:.2%}")
        print(f"年化波动率: {volatility:.2%}")
        print(f"夏普比率: {sharpe_ratio:.2f}")
        print(f"最大回撤: {max_drawdown:.2%}")
        print(f"胜率: {win_rate:.2%}")
        
        # 绘制策略收益曲线
        self._plot_strategy_performance(strategy_returns, test_df.index[1:len(strategy_returns)+1])
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }
    
    def _plot_strategy_performance(self, strategy_returns, dates):
        """绘制策略性能图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 累积收益曲线
        cumulative_returns = np.cumprod(1 + strategy_returns)
        axes[0, 0].plot(dates, cumulative_returns, label='Strategy', linewidth=2)
        axes[0, 0].set_title('Cumulative Returns')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 回撤曲线
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        axes[0, 1].fill_between(dates, drawdown, 0, alpha=0.3, color='red')
        axes[0, 1].plot(dates, drawdown, color='red', linewidth=1)
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown')
        axes[0, 1].grid(True)
        
        # 收益分布
        axes[1, 0].hist(strategy_returns, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Return Distribution')
        axes[1, 0].set_xlabel('Daily Return')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True)
        
        # 情绪得分时间序列
        if self.sentiment_scores is not None:
            sentiment_subset = self.sentiment_scores.loc[dates]
            axes[1, 1].plot(dates, sentiment_subset, color='green', linewidth=1)
            axes[1, 1].set_title('Sentiment Scores')
            axes[1, 1].set_ylabel('Sentiment Score')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance(self):
        """获取特征重要性"""
        if self.sentiment_scores is None:
            print("请先训练模型!")
            return None
        
        # 计算特征与情绪得分的相关性
        correlations = {}
        for col in self.feature_cols:
            if col in self.factors_df.columns:
                corr = np.corrcoef(self.factors_df[col].dropna(), 
                                 self.sentiment_scores.dropna())[0, 1]
                correlations[col] = abs(corr)
        
        # 排序并返回
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        print("\n=== 特征重要性 (基于与情绪得分的相关性) ===")
        for feature, importance in sorted_features[:10]:
            print(f"{feature}: {importance:.4f}")
        
        return correlations
