#优化后代码 （解决了数据加载问题+优化了模型）

# 安装必要库并设置中文字体（本地环境解决方案）
# 注意：本地环境不需要安装字体，使用系统默认字体即可

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import gc
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from scipy.stats import spearmanr, kendalltau
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score
# from google.colab import drive  # 注释掉Google Colab依赖
import warnings
import matplotlib.font_manager as fm
import logging
from contextlib import closing


# 设置中文字体（本地环境解决方案）
try:
    # 尝试使用系统中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    print("中文字体设置成功")
except Exception as e:
    print(f"中文字体设置失败: {e}")
    print("将使用默认字体，中文可能显示为方块")


warnings.filterwarnings('ignore')
# ====================== 数据准备模块 ======================
class DataLoader:
    def __init__(self):
        # Mount Google Drive (Colab environment)
        # drive.mount('/content/drive', force_remount=True) # 注释掉Google Colab依赖

        # Use user-provided file paths (modify to your actual paths)
        # The paths here are what you specified; please ensure these files are in your Google Drive and the paths are correct
        # 使用相对路径，假设数据文件在上级目录的data文件夹中
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(current_dir), 'data')
        
        self.pct_path = os.path.join(data_dir, '主板涨跌幅数据.xlsx')
        self.open_path = os.path.join(data_dir, '主板开盘价数据.xlsx')
        self.close_path = os.path.join(data_dir, '主板收盘价数据.xlsx')
        self.pct_df = None
        self.open_df = None
        self.close_df = None

    def load_data(self):
        """加载并预处理数据（适配Colab环境）"""
        print("正在加载数据...")
        try:
            # Read Excel files
            # Use engine='openpyxl' for compatibility
            # Assuming '日期' column is the index
            self.pct_df = pd.read_excel(self.pct_path, index_col='日期', engine='openpyxl')
            self.open_df = pd.read_excel(self.open_path, index_col='日期', engine='openpyxl')
            self.close_df = pd.read_excel(self.close_path, index_col='日期', engine='openpyxl')

            print("原始数据加载完成。")

            # Ensure date index is datetime type, coercing errors to NaT
            print("尝试将索引转换为datetime并处理无法解析的日期...")
            self.pct_df.index = pd.to_datetime(self.pct_df.index, errors='coerce')
            self.open_df.index = pd.to_datetime(self.open_df.index, errors='coerce')
            self.close_df.index = pd.to_datetime(self.close_df.index, errors='coerce')


            # --- 检查转换后是否有NaT (添加调试信息) ---
            nat_count_pct = self.pct_df.index.isna().sum()
            if nat_count_pct > 0:
                 print(f"警告: pct_df 索引中发现 {nat_count_pct} 个无法解析的日期 (NaT)。这些行可能在后续清洗中被移除。")
            # --------------------------------------------


            # Data Cleaning
            print("\n开始数据清洗...")
            # Pass all dataframes to the cleaning method to ensure consistency
            cleaned_pct_df, cleaned_open_df, cleaned_close_df = self._clean_data(self.pct_df, self.open_df, self.close_df)
            self.pct_df = cleaned_pct_df
            self.open_df = cleaned_open_df
            self.close_df = cleaned_close_df

            print("数据清洗完成.")

            # --- Check final data range ---
            if self.pct_df is not None and not self.pct_df.empty:
                # Use .strftime('%Y-%m-%d') for clean date string display
                print(f"\n数据加载和清洗完成，最终时间范围: {self.pct_df.index[0].strftime('%Y-%m-%d')} 至 {self.pct_df.index[-1].strftime('%Y-%m-%d')}")
                print(f"最终股票数量: {len(self.pct_df.columns)}")
                print(f"清洗后 pct_df shape: {self.pct_df.shape}")
                print(f"清洗后 NaNs in pct_df: {self.pct_df.isnull().sum().sum()}")
            else:
                 print("\n数据清洗后 DataFrame 为空，请检查数据和清洗逻辑。")


            return self.pct_df, self.open_df, self.close_df

        except FileNotFoundError as e:
            print(f"数据加载失败: 文件未找到 - {e}")
            print("请检查文件路径是否正确。")
            return None, None, None
        except Exception as e:
            print(f"数据加载失败: {str(e)}")
            # Print more detailed error information for debugging
            import traceback
            traceback.print_exc()
            return None, None, None


    def _clean_data(self, pct_df, open_df, close_df):
        """数据清洗 - 仅移除缺失多的个股并进行前向填充"""

        # 1. Check and convert percentage values (apply to pct_df)
        # If max value is less than 1.5, assume it's in decimal form (e.g., 0.01 for 1%) and convert to percentage by multiplying by 100.
        # The threshold 1.5 might need adjustment based on your actual data.
        # Note: This only applies to pct_df as open/close are prices.
        if pct_df.max().max() < 1.5:
            print("检测到涨跌幅为小数形式，转换为百分比...")
            pct_df = pct_df * 100  # Convert to percentage form


        # 2. Remove columns (stocks) with more than 50% missing data across the entire period
        # This step prioritizes keeping the historical dates by removing problematic stocks
        print("移除缺失数据超过 50% 的个股...")
        initial_stock_count = pct_df.shape[1]
        missing_percentage_per_stock = pct_df.isnull().mean()
        # Define threshold for removing stocks (e.g., > 50% missing)
        stock_missing_threshold = 0.5
        stocks_to_keep = missing_percentage_per_stock[missing_percentage_per_stock <= stock_missing_threshold].index

        pct_df = pct_df[stocks_to_keep]
        open_df = open_df[stocks_to_keep]
        close_df = close_df[stocks_to_keep]
        stocks_removed_count = initial_stock_count - pct_df.shape[1]
        if stocks_removed_count > 0:
            print(f"移除了 {stocks_removed_count} 只个股，因其缺失数据超过 {stock_missing_threshold:.0%}。")
        else:
            print("没有个股因缺失数据过多而被移除。")


        # 3. Forward fill missing values within the remaining stocks/dates
        # This is done after removing stocks with excessive missing data
        print("进行前向填充缺失值...")
        pct_df = pct_df.ffill(axis=0)
        open_df = open_df.ffill(axis=0)
        close_df = close_df.ffill(axis=0)

        # Note: We are explicitly NOT removing rows based on stock coverage threshold here
        # Any rows with NaT index from to_datetime(errors='coerce') will remain unless they
        # became entirely NaN after stock removal and ffill (unlikely).
        # If you need to remove rows with NaT index, add .dropna(axis=0, how='all') here.
        # For now, assuming the goal is to keep all original dates possible.


        # 4. Remove any columns that became entirely NaN after ffill (shouldn't happen but as a safeguard)
        initial_stock_count_post_ffill = pct_df.shape[1]
        pct_df = pct_df.dropna(axis=1, how='all')
        open_df = open_df.dropna(axis=1, how='all')
        close_df = close_df.dropna(axis=1, how='all')
        cols_removed_post_ffill = initial_stock_count_post_ffill - pct_df.shape[1]
        if cols_removed_post_ffill > 0:
             print(f"移除了 {cols_removed_post_ffill} 列，因其在清洗后仍然完全缺失。")


        # Final check for NaNs (should only be leading NaNs that ffill couldn't handle)
        print(f"清洗后 NaNs in pct_df: {pct_df.isnull().sum().sum()}")


        return pct_df, open_df, close_df

        # 添加 close 方法
    def close(self):
        """空方法，满足上下文管理器要求"""
        pass
# ====================== 因子计算模块 ======================
class FactorCalculator:
    def __init__(self, pct_df, open_df, close_df):
        self.pct_df = pct_df
        self.open_df = open_df
        self.close_df = close_df
        self.factors_df = pd.DataFrame(index=pct_df.index)
        self.UP_LIMIT = 9.9  # 涨停阈值
        self.DOWN_LIMIT = -9.9  # 跌停阈值
        self.EXTREME_RETURN_THRESHOLD = 50  # 极端收益阈值(百分比)

    def calculate_factors(self):
        """计算市场情绪因子（增强鲁棒性）"""
        print("开始计算市场情绪因子...")

        # 1. 计算每日涨停/跌停股票数
        limit_up_mask = (self.pct_df >= self.UP_LIMIT)
        limit_down_mask = (self.pct_df <= self.DOWN_LIMIT)

        limit_up_counts = limit_up_mask.sum(axis=1)
        limit_down_counts = limit_down_mask.sum(axis=1)
        total_stocks = self.pct_df.count(axis=1)

        # 2. 涨停板占比因子
        self.factors_df['涨停占比'] = limit_up_counts / total_stocks

        # 3. 跌停板占比因子
        self.factors_df['跌停占比'] = limit_down_counts / total_stocks

        # 4. 净涨停占比因子
        self.factors_df['净涨停占比'] = (limit_up_counts - limit_down_counts) / total_stocks

        # 5. 打板策略收益因子（涨停次日收益）
        next_day_returns = self.pct_df.shift(-1)

        # 优化打板收益因子计算（添加异常值过滤）
        print("优化打板收益因子计算（添加异常值过滤）...")
        limit_up_next_returns = next_day_returns.where(limit_up_mask)

        # 应用缩尾处理过滤极端值
        if limit_up_next_returns.abs().max().max() > self.EXTREME_RETURN_THRESHOLD:
            print(f"检测到极端打板收益值，应用缩尾处理...")
            lower = limit_up_next_returns.quantile(0.01)
            upper = limit_up_next_returns.quantile(0.99)
            limit_up_next_returns = limit_up_next_returns.clip(lower, upper)

        self.factors_df['打板收益'] = limit_up_next_returns.mean(axis=1)

        # 6. 跌停次日收益因子
        limit_down_next_returns = next_day_returns.where(limit_down_mask)
        if limit_down_next_returns.abs().max().max() > self.EXTREME_RETURN_THRESHOLD:
            print(f"检测到极端跌停收益值，应用缩尾处理...")
            lower = limit_down_next_returns.quantile(0.01)
            upper = limit_down_next_returns.quantile(0.99)
            limit_down_next_returns = limit_down_next_returns.clip(lower, upper)
        self.factors_df['跌停收益'] = limit_down_next_returns.mean(axis=1)

        # 7. 市场平均收益
        self.factors_df['市场收益'] = self.pct_df.mean(axis=1)

        # 填充可能的缺失值
        self.factors_df = self.factors_df.ffill().dropna()

        print("因子计算完成!")

        # 添加滞后因子
        for factor in ['涨停占比', '跌停占比', '净涨停占比', '打板收益', '跌停收益']:
            self.factors_df[f'{factor}_lag1'] = self.factors_df[factor].shift(1)
            self.factors_df[f'{factor}_lag2'] = self.factors_df[factor].shift(2)

        # 移除因滞后产生的缺失值
        self.factors_df = self.factors_df.dropna()

        return self.factors_df

# ====================== 因子分析模块 ======================
class FactorAnalyzer:
    def __init__(self, factors_df):
        self.factors_df = factors_df

    def calculate_global_ic(self):
        """计算因子全局IC值（信息系数）"""
        print("计算因子全局IC...")
        ic_results = {}
        ic_summary = pd.DataFrame(columns=['IC均值', 'IC标准差', 'IR', 'IC>0比例', 'p值'])

        target = self.factors_df['市场收益'].shift(-1).dropna()
        ic_factors = self.factors_df.drop(columns=['市场收益'] + [col for col in self.factors_df.columns if '_lag' in col])
        ic_factors = ic_factors.dropna()

        for factor in ic_factors.columns:
            factor_vals = ic_factors[factor].dropna()
            common_idx = factor_vals.index.intersection(target.index)

            if len(common_idx) == 0:
                continue

            ic, p_value = spearmanr(factor_vals.loc[common_idx], target.loc[common_idx])
            ic_results[factor] = ic
            ic_summary.loc[factor] = [
                ic,
                np.nan,
                np.nan,
                1 if ic > 0 else 0,
                p_value
            ]

        # 可视化IC结果
        self.plot_ic_results(ic_results)

        return ic_results, ic_summary

    def calculate_rolling_ic(self, window=252):
        """计算滚动IC（时序IC）"""
        print("计算滚动IC...")
        target = self.factors_df['市场收益'].shift(-1).dropna()
        ic_factors = self.factors_df.drop(columns=['市场收益'] + [col for col in self.factors_df.columns if '_lag' in col])

        rolling_ic = pd.DataFrame(index=ic_factors.index)

        for factor in ic_factors.columns:
            factor_vals = ic_factors[factor]
            ic_series = []

            for i in range(window, len(factor_vals)):
                start = i - window
                end = i

                # 计算窗口期内的Spearman相关系数
                ic, _ = spearmanr(factor_vals.iloc[start:end], target.iloc[start:end])
                ic_series.append(ic)

            # 对齐索引
            rolling_ic[factor] = pd.Series(ic_series, index=factor_vals.index[window:])

        # 可视化滚动IC
        plt.figure(figsize=(14, 8))
        for factor in rolling_ic.columns:
            rolling_ic[factor].plot(label=factor, alpha=0.7)

        plt.title(f'滚动IC (窗口={window}天)', fontsize=14)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('IC值', fontsize=12)
        plt.axhline(y=0.05, color='green', linestyle='--', alpha=0.5)
        plt.axhline(y=-0.05, color='green', linestyle='--', alpha=0.5)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # 计算IR (信息比率)
        ic_summary = pd.DataFrame()
        for factor in rolling_ic.columns:
            mean_ic = rolling_ic[factor].mean()
            std_ic = rolling_ic[factor].std()
            ir = mean_ic / std_ic if std_ic != 0 else 0

            ic_summary.loc[factor, 'IC均值'] = mean_ic
            ic_summary.loc[factor, 'IC标准差'] = std_ic
            ic_summary.loc[factor, 'IR'] = ir
            ic_summary.loc[factor, 'IC>0比例'] = (rolling_ic[factor] > 0).mean()

        print("\n时序IC统计摘要:")
        return rolling_ic, ic_summary

    def plot_ic_results(self, ic_results):
        """可视化IC分析结果"""
        factors = list(ic_results.keys())
        ic_values = list(ic_results.values())

        plt.figure(figsize=(12, 6))
        colors = ['green' if val > 0 else 'red' for val in ic_values]
        bars = plt.bar(factors, ic_values, color=colors, alpha=0.7)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10)

        plt.xlabel('因子', fontsize=12)
        plt.ylabel('IC值', fontsize=12)
        plt.title('因子信息系数(IC)分析', fontsize=14)
        plt.axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label='显著阈值(0.05)')
        plt.axhline(y=-0.05, color='green', linestyle='--', alpha=0.5)
        plt.legend(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        print("\n" + "="*80)
        print("因子IC解释:")
        print("IC(Information Coefficient)衡量因子预测能力:")
        print("- 正IC: 因子值越高，次日市场收益越高")
        print("- 负IC: 因子值越高，次日市场收益越低")
        print("- |IC| > 0.05 通常被认为具有预测能力")
        print("="*80)

    def analyze_factor_relationships(self):
        """分析因子间相关性"""
        corr_factors = self.factors_df.drop(columns=[col for col in self.factors_df.columns if '_lag' in col])
        corr_matrix = corr_factors.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", center=0,
                   annot_kws={"size": 10}, cbar_kws={"shrink": .8})
        plt.title("因子间相关系数矩阵", fontsize=14)
        plt.xticks(fontsize=10, rotation=45)
        plt.yticks(fontsize=10, rotation=0)
        plt.tight_layout()
        plt.show()

        return corr_matrix

    def plot_factor_ts(self):
        """绘制因子时间序列"""
        ts_factors = self.factors_df.drop(columns=[col for col in self.factors_df.columns if '_lag' in col])
        plt.figure(figsize=(15, len(ts_factors.columns)*2))
        for i, factor in enumerate(ts_factors.columns):
            plt.subplot(len(ts_factors.columns), 1, i+1)
            ts_factors[factor].plot(title=f'{factor}时间序列', color='royalblue', linewidth=1)
            plt.axhline(y=ts_factors[factor].mean(), color='r', linestyle='--', alpha=0.5)
            plt.ylabel('因子值', fontsize=10)
            plt.grid(axis='y', linestyle='--', alpha=0.5)
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.show()

# ====================== 情绪模型模块 ======================
class SentimentModel:
    def __init__(self, factors_df):
        self.factors_df = factors_df
        self.scaler = StandardScaler()
        self.models = {
            '线性回归': LinearRegression(),
            '岭回归': Ridge(alpha=1.0),
            '随机森林': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5),
            '梯度提升': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=3),
            'LSTM': self.build_lstm_model()
        }
        self.best_model = None
        self.best_score = -np.inf
        self.weights = None
        self.sentiment_scores = None

    def build_lstm_model(self):
        """构建LSTM模型"""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout

        model = Sequential()
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(32))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

    def create_sequences(self, X, y, seq_length):
        """创建时间序列数据"""
        X_seq, y_seq = [], []
        for i in range(seq_length, len(X)):
            X_seq.append(X[i-seq_length:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    def train_and_compare_models(self, test_size=0.2):
        """训练并比较多种模型（添加特征选择）"""
        print("训练并比较情绪预测模型...")

        # 准备数据
        X_cols = [col for col in self.factors_df.columns if col != '打板收益' and '_lag' in col] + \
                 ['涨停占比', '跌停占比', '净涨停占比', '跌停收益']
        y_col = '打板收益'

        X = self.factors_df[X_cols]
        y = self.factors_df[y_col]

        combined = pd.concat([X, y], axis=1).dropna()
        X = combined[X_cols]
        y = combined[y_col]

        # 划分训练集和测试集
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # 标准化数据
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 特征选择（递归特征消除）
        print("进行递归特征消除(RFE)选择重要特征...")
        tscv = TimeSeriesSplit(n_splits=5)
        selector = RFECV(
            estimator=LinearRegression(),
            step=1,
            cv=tscv,
            scoring='r2',
            min_features_to_select=5
        )
        selector.fit(X_train_scaled, y_train)
        selected_features = np.array(X_cols)[selector.support_]
        print(f"选择的特征({len(selected_features)}个): {', '.join(selected_features)}")

        # 使用选择的特征
        X_train_selected = X_train_scaled[:, selector.support_]
        X_test_selected = X_test_scaled[:, selector.support_]

        # 存储模型性能结果
        model_performance = {}
        tscv = TimeSeriesSplit(n_splits=5)

        plt.figure(figsize=(14, len(self.models)*4))

        # 准备LSTM数据
        seq_length = 10
        X_train_lstm, y_train_lstm = self.create_sequences(X_train_scaled, y_train, seq_length)
        X_test_lstm, y_test_lstm = self.create_sequences(X_test_scaled, y_test, seq_length)

        for i, (model_name, model) in enumerate(self.models.items()):
            print(f"训练模型: {model_name}")

            if model_name == 'LSTM':
                # 训练LSTM模型
                history = model.fit(
                    X_train_lstm, y_train_lstm,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_test_lstm, y_test_lstm),
                    verbose=0
                )
                y_pred = model.predict(X_test_lstm).flatten()
                test_score = r2_score(y_test_lstm, y_pred)
                cv_score = np.nan  # LSTM不适合交叉验证
            else:
                # 训练其他模型
                cv_scores = cross_val_score(model, X_train_selected, y_train,
                                          cv=tscv, scoring='r2')
                mean_cv_score = np.mean(cv_scores)
                model.fit(X_train_selected, y_train)
                y_pred = model.predict(X_test_selected)
                test_score = r2_score(y_test, y_pred)

            # 存储性能
            model_performance[model_name] = {
                'cv_score': mean_cv_score if model_name != 'LSTM' else np.nan,
                'test_score': test_score
            }

            # 可视化预测效果
            plt.subplot(len(self.models), 2, i*2+1)
            if model_name == 'LSTM':
                actual = y_test_lstm
            else:
                actual = y_test

            plt.scatter(actual, y_pred, alpha=0.5, s=20)
            min_val = min(actual.min(), y_pred.min())
            max_val = max(actual.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
            plt.xlabel('实际值', fontsize=10)
            plt.ylabel('预测值', fontsize=10)
            title = f'{model_name}预测效果\n'
            if model_name != 'LSTM':
                title += f'CV R²: {mean_cv_score:.4f}\n'
            title += f'Test R²: {test_score:.4f}'
            plt.title(title, fontsize=12)
            plt.grid(True, alpha=0.3)

            # 可视化特征重要性/权重
            plt.subplot(len(self.models), 2, i*2+2)
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = model.coef_
            else:
                importances = np.zeros(len(selected_features))

            if np.sum(np.abs(importances)) > 0:
                features = selected_features
                sorted_idx = np.argsort(np.abs(importances))[::-1]
                plt.bar(range(len(importances)), importances[sorted_idx], color='skyblue')
                plt.xticks(range(len(importances)), [features[i] for i in sorted_idx], rotation=45, fontsize=10)
                plt.title(f'{model_name}特征重要性/权重', fontsize=12)
                plt.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 选择最佳模型
        self.best_model_name = max(model_performance,
                                  key=lambda x: model_performance[x]['test_score'])
        self.best_model = self.models[self.best_model_name]
        self.best_score = model_performance[self.best_model_name]['test_score']

        print(f"\n最佳模型: {self.best_model_name}, 测试集R²: {self.best_score:.4f}")

        # 使用全部数据重新训练最佳模型
        X_full = self.factors_df[X_cols]
        y_full = self.factors_df[y_col]
        combined_full = pd.concat([X_full, y_full], axis=1).dropna()
        X_full_aligned = combined_full[X_cols]
        y_full_aligned = combined_full[y_col]

        X_full_scaled = self.scaler.transform(X_full_aligned)
        if self.best_model_name == 'LSTM':
            X_full_lstm, y_full_lstm = self.create_sequences(X_full_scaled, y_full_aligned, seq_length)
            self.best_model.fit(X_full_lstm, y_full_lstm)
        else:
            X_full_selected = X_full_scaled[:, selector.support_]
            self.best_model.fit(X_full_selected, y_full_aligned)

        # 计算特征权重
        self.calculate_feature_weights(selected_features)

        # 计算情绪得分
        if self.best_model_name == 'LSTM':
            X_all_lstm, _ = self.create_sequences(self.scaler.transform(self.factors_df[X_cols]),
                                               self.factors_df[y_col], seq_length)
            sentiment_scores = self.best_model.predict(X_all_lstm).flatten()
            # 对齐索引
            sentiment_scores = pd.Series(sentiment_scores,
                                        index=self.factors_df.index[seq_length:],
                                        name='情绪得分')
        else:
            X_all_selected = self.scaler.transform(self.factors_df[X_cols])[:, selector.support_]
            sentiment_scores = self.best_model.predict(X_all_selected)
            sentiment_scores = pd.Series(sentiment_scores,
                                        index=self.factors_df.index,
                                        name='情绪得分')

        self.sentiment_scores = sentiment_scores.dropna()

        return model_performance

    def calculate_feature_weights(self, feature_names):
        """计算特征权重（不同模型不同方法）"""
        if isinstance(self.best_model, (LinearRegression, Ridge)):
            self.weights = dict(zip(feature_names, self.best_model.coef_))
        elif hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            self.weights = dict(zip(feature_names, importances))
        else:
            self.weights = {factor: 1/len(feature_names) for factor in feature_names}
        return self.weights

    def plot_weights(self):
        """可视化因子权重"""
        if self.weights is None:
            print("请先训练模型!")
            return

        plt.figure(figsize=(12, 7))
        factors = list(self.weights.keys())
        weights = list(self.weights.values())

        sorted_idx = np.argsort(np.abs(weights))[::-1]
        factors = [factors[i] for i in sorted_idx]
        weights = [weights[i] for i in sorted_idx]

        colors = ['green' if w > 0 else 'red' for w in weights] if isinstance(self.best_model, (LinearRegression, Ridge)) else 'skyblue'

        bars = plt.bar(factors, weights, color=colors, alpha=0.7)

        for bar in bars:
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va=va, fontsize=9)

        plt.title(f'{self.best_model_name}模型因子权重/重要性', fontsize=14)
        plt.ylabel('权重 / 重要性', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

# ====================== 决策引擎模块 ======================
class TradingDecisionEngine:
    def __init__(self, factors_df, sentiment_scores):
        self.factors_df = factors_df
        self.sentiment_scores = sentiment_scores
        common_index = self.factors_df.index.intersection(self.sentiment_scores.index)
        self.factors_df_aligned = self.factors_df.loc[common_index]
        self.sentiment_scores_aligned = self.sentiment_scores.loc[common_index]
        self.hist_scores = self.sentiment_scores_aligned.values
        self.quantiles = self.calculate_dynamic_quantiles()

    def calculate_dynamic_quantiles(self):
        """计算情绪得分的动态分位数阈值"""
        scores = self.sentiment_scores_aligned
        return {
            '极度乐观': scores.quantile(0.95),
            '乐观': scores.quantile(0.75),
            '中性偏多': scores.quantile(0.55),
            '中性': scores.quantile(0.45),
            '中性偏空': scores.quantile(0.25),
            '悲观': scores.quantile(0.05)
        }

    def get_sentiment_level(self, score):
        """使用动态分位数划分情绪级别"""
        q = self.quantiles
        if score > q['极度乐观']:
            return "极度乐观"
        elif score > q['乐观']:
            return "乐观"
        elif score > q['中性偏多']:
            return "中性偏多"
        elif score > q['中性']:
            return "中性"
        elif score > q['中性偏空']:
            return "中性偏空"
        elif score > q['悲观']:
            return "悲观"
        else:
            return "极度悲观"

    def calculate_market_volatility(self, date):
        """计算市场波动率（基于前5日市场收益标准差）"""
        start_date = date - pd.Timedelta(days=7)
        market_returns = self.factors_df_aligned.loc[start_date:date, '市场收益']
        if len(market_returns) < 3:
            return 0.0
        daily_std = market_returns.std()
        annualized_vol = daily_std * np.sqrt(252)
        return annualized_vol / 100  # 转换为小数形式

    def make_decision(self, date_input, position_type='holding'):
        """生成交易决策（添加风险控制）"""
        try:
            date = pd.to_datetime(date_input)
        except:
            return {"error": "无效的日期格式"}

        if date not in self.sentiment_scores_aligned.index:
            return {"error": f"日期 {date.strftime('%Y-%m-%d')} 不在数据范围内"}

        sentiment = self.sentiment_scores_aligned.loc[date]
        sentiment_level = self.get_sentiment_level(sentiment)
        percentile = np.sum(self.hist_scores < sentiment) / len(self.hist_scores) if len(self.hist_scores) > 0 else 0

        limit_up_ratio = self.factors_df_aligned.loc[date, '涨停占比']
        net_limit_ratio = self.factors_df_aligned.loc[date, '净涨停占比']
        limit_up_return = self.factors_df_aligned.loc[date, '打板收益']
        limit_down_ratio = self.factors_df_aligned.loc[date, '跌停占比']
        limit_down_return = self.factors_df_aligned.loc[date, '跌停收益']

        if pd.isna(limit_up_ratio) or pd.isna(net_limit_ratio) or pd.isna(limit_up_return):
            return {"error": f"日期 {date.strftime('%Y-%m-%d')} 存在缺失的市场指标数据"}

        # 计算市场波动率
        volatility = self.calculate_market_volatility(date)

        if position_type == 'holding':
            # 持仓股涨停次日决策（动态阈值）
            if sentiment > self.quantiles['极度乐观']:
                action = "继续持有"
                reason = f"市场情绪{sentiment_level} (得分: {sentiment:.2f}，分位数: {percentile:.1%})，预测涨停股有较高溢价概率"
            elif sentiment > self.quantiles['乐观']:
                action = "部分止盈(50%)"
                reason = f"市场情绪{sentiment_level} (得分: {sentiment:.2f}，分位数: {percentile:.1%})，当日涨停股占比{limit_up_ratio:.2%}，建议锁定部分利润"
            elif sentiment > self.quantiles['中性']:
                action = "开盘卖出"
                reason = f"市场情绪{sentiment_level} (得分: {sentiment:.2f}，分位数: {percentile:.1%})，当日净涨停占比{net_limit_ratio:.2%}，当日涨停股溢价概率低"
            else:
                action = "立即卖出"
                reason = f"市场情绪{sentiment_level} (得分: {sentiment:.2f}，分位数: {percentile:.1%})，市场存在恐慌情绪，建议立即避险"
        else:
            # 自选股追涨决策（动态阈值）
            if sentiment > self.quantiles['极度乐观']:
                action = "集合竞价追涨"
                reason = f"市场情绪{sentiment_level} (得分: {sentiment:.2f}，分位数: {percentile:.1%})，预测涨停股溢价高"
            elif sentiment > self.quantiles['乐观']:
                action = "开盘回踩追涨"
                reason = f"市场情绪{sentiment_level} (得分: {sentiment:.2f}，分位数: {percentile:.1%})，可逢低参与"
            else:
                action = "不追涨"
                reason = f"市场情绪{sentiment_level} (得分: {sentiment:.2f}，分位数: {percentile:.1%})，涨停股溢价低或存在风险"

        # 添加风险管理规则
        if volatility > 0.03:  # 高波动率市场
            reason += "。市场波动剧烈，建议降低仓位"
            if "追涨" in action:
                action = "谨慎" + action
            elif "持有" in action and "部分止盈" not in action:
                action = "考虑部分止盈"

        # 添加止损规则（简化示例）
        if position_type == 'holding' and "持有" in action:
            # 假设我们有持仓成本信息（实际应用中需要维护持仓记录）
            holding_days = 1  # 简化处理
            if holding_days > 3 and sentiment < 0:
                action = "考虑止损"
                reason += "。持仓超过3天且情绪转负，建议止损"

        return {
            "date": date.strftime('%Y-%m-%d'),
            "sentiment_score": round(sentiment, 4),
            "sentiment_level": sentiment_level,
            "percentile": round(percentile, 4),
            "action": action,
            "reason": reason,
            "volatility": round(volatility, 4),
            "indicators": {
                "涨停占比": round(limit_up_ratio, 4),
                "跌停占比": round(limit_down_ratio, 4),
                "净涨停占比": round(net_limit_ratio, 4),
                "当日打板收益(因子值)": round(limit_up_return, 4),
                "当日跌停收益(因子值)": round(limit_down_return, 4)
            }
        }

# ====================== 回测模块 ======================
class StrategyBacktester:
    def __init__(self, factors_df, sentiment_scores, initial_capital=1000000):
        self.factors_df = factors_df
        self.sentiment_scores = sentiment_scores
        self.initial_capital = initial_capital
        self.portfolio = pd.DataFrame(index=factors_df.index, columns=['value', 'position'])
        self.trade_history = pd.DataFrame(columns=[
            'date', 'action', 'position', 'market_return',
            'portfolio_value', 'commission'
        ])
        self.commission_rate = 0.00025  # 交易佣金0.025%
        self.stamp_tax = 0.001  # 印花税0.1%

    def run_backtest(self):
        """运行回测（添加交易成本）"""
        print("开始策略回测...")
        capital = self.initial_capital
        position = 0

        common_index = self.sentiment_scores.index.intersection(self.factors_df.index)
        factors_df_for_backtest_dates = self.factors_df.loc[common_index].shift(-1).dropna()
        sentiment_scores_aligned = self.sentiment_scores.loc[factors_df_for_backtest_dates.index]
        valid_dates = factors_df_for_backtest_dates.index

        if len(valid_dates) == 0:
            print("错误：没有足够的数据进行回测（情绪得分和市场收益无法对齐）。")
            return None, None

        # 基准收益
        benchmark = (1 + self.factors_df.loc[valid_dates, '市场收益']/100).cumprod() * self.initial_capital

        # 初始化投资组合
        first_date = valid_dates[0]
        self.portfolio.loc[first_date, 'value'] = self.initial_capital
        self.portfolio.loc[first_date, 'position'] = 0

        # 回测循环
        for i in range(1, len(valid_dates)):
            date = valid_dates[i]
            prev_date = valid_dates[i-1]

            # 获取前一日情绪得分
            prev_sentiment = sentiment_scores_aligned.loc[prev_date]

            # 确定目标仓位（基于情绪分位数）
            q = self.sentiment_scores.quantile
            if prev_sentiment > q(0.95):
                target_position = 1.0
            elif prev_sentiment > q(0.75):
                target_position = 0.8
            elif prev_sentiment > q(0.50):
                target_position = 0.6
            elif prev_sentiment > q(0.25):
                target_position = 0.4
            elif prev_sentiment > q(0.05):
                target_position = 0.2
            else:
                target_position = 0.0

            # 平滑仓位调整
            prev_day_position = self.portfolio.loc[prev_date, 'position'] if prev_date in self.portfolio.index else 0
            current_day_smoothed_position = 0.7 * prev_day_position + 0.3 * target_position

            # 计算仓位变化
            position_change = current_day_smoothed_position - prev_day_position

            # 计算交易成本
            commission = 0
            if position_change != 0:
                trade_amount = abs(position_change) * self.portfolio.loc[prev_date, 'value']
                commission = trade_amount * self.commission_rate
                if position_change < 0:  # 卖出操作
                    commission += trade_amount * self.stamp_tax

            # 获取当日市场收益
            market_return = self.factors_df.loc[date, '市场收益'] / 100

            # 计算当日收益
            daily_return = current_day_smoothed_position * market_return

            # 更新资金（考虑交易成本）
            capital = capital * (1 + daily_return) - commission

            # 记录投资组合
            self.portfolio.loc[date, 'value'] = capital
            self.portfolio.loc[date, 'position'] = current_day_smoothed_position

            # 记录交易历史
            trade_record = {
                'date': date,
                'action': '增仓' if position_change > 0 else '减仓' if position_change < 0 else '保持',
                'position': current_day_smoothed_position,
                'market_return': market_return,
                'portfolio_value': capital,
                'commission': commission
            }
            self.trade_history = pd.concat(
                [self.trade_history, pd.DataFrame([trade_record])],
                ignore_index=True
            )

        # 确保投资组合索引对齐
        self.portfolio = self.portfolio.loc[valid_dates].dropna()
        benchmark = benchmark.loc[self.portfolio.index]

        print("回测完成!")
        return self.portfolio, benchmark

    def calculate_performance(self, benchmark_returns=None):
        """计算策略绩效（添加更多指标）"""
        if self.portfolio is None or self.portfolio.empty:
            print("回测未成功运行或没有足够的有效数据计算绩效!")
            return None

        portfolio = self.portfolio['value']
        returns = portfolio.pct_change().dropna()

        if returns.empty:
            print("没有足够的交易日计算收益。")
            return {
                "累计收益(%)": 0.0,
                "年化收益(%)": 0.0,
                "最大回撤(%)": 0.0,
                "夏普比率": 0.0,
                "胜率(%)": 0.0
            }

        # 基础绩效指标
        cumulative_return = (portfolio.iloc[-1] / portfolio.iloc[0] - 1) * 100
        years = (portfolio.index[-1] - portfolio.index[0]).days / 365.25
        annualized_return = (1 + cumulative_return/100) ** (1/years) - 1 if years > 0 else 0

        peak = portfolio.cummax()
        drawdown = (portfolio - peak) / peak
        max_drawdown = drawdown.min() * 100 if not drawdown.empty else 0

        sharpe_std = returns.std()
        sharpe_ratio = returns.mean() / sharpe_std * np.sqrt(252) if sharpe_std > 0 else 0

        win_rate = (returns > 0).mean() * 100

        # 高级绩效指标
        calmar_ratio = annualized_return / (abs(max_drawdown) / 100) if max_drawdown != 0 else 0
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        sortino_ratio = returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0
        trade_freq = (self.trade_history['action'] != '保持').mean() * 100

        # 基准比较（如果提供）
        excess_return = cumulative_return
        info_ratio = np.nan
        if benchmark_returns is not None:
            bench_returns = benchmark_returns.pct_change().dropna()
            bench_cumulative = (bench_returns + 1).prod() - 1
            excess_return = cumulative_return - bench_cumulative * 100
            active_returns = returns - bench_returns
            info_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252)

        return {
            "累计收益(%)": cumulative_return,
            "年化收益(%)": annualized_return * 100,
            "最大回撤(%)": max_drawdown,
            "夏普比率": sharpe_ratio,
            "胜率(%)": win_rate,
            "卡玛比率": calmar_ratio,
            "索提诺比率": sortino_ratio,
            "交易频率(%)": trade_freq,
            "平均仓位(%)": self.portfolio['position'].mean() * 100,
            "超额收益(%)": excess_return,
            "信息比率": info_ratio
        }

    def plot_results(self, benchmark):
        """可视化回测结果（增强可视化）"""
        if self.portfolio is None or self.portfolio.empty:
            print("回测结果数据为空，无法绘制图表。")
            return

        plt.figure(figsize=(14, 12))

        # 净值曲线
        plt.subplot(3, 1, 1)
        portfolio_normalized = self.portfolio['value'] / self.initial_capital
        benchmark_normalized = benchmark / benchmark.iloc[0] if not benchmark.empty else pd.Series([1], index=[self.portfolio.index[0]])

        portfolio_normalized.plot(label='策略净值', color='darkgreen', linewidth=1.5)
        if not benchmark.empty:
            benchmark_normalized.plot(label='基准净值', color='blue', alpha=0.7, linewidth=1.5)

        plt.title('策略净值曲线', fontsize=14)
        plt.ylabel('净值', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # 回撤曲线
        plt.subplot(3, 1, 2)
        peak = portfolio_normalized.cummax()
        drawdown = (portfolio_normalized - peak) / peak
        drawdown.plot(color='red', alpha=0.7)
      # [新增代码]：强制将 drawdown 转换为数值类型，无法转换的值设为 NaN
        drawdown_numeric = pd.to_numeric(drawdown, errors='coerce')

    # [修改代码]：使用清洗后的 drawdown_numeric 进行绘图
        plt.fill_between(drawdown_numeric.index, drawdown_numeric, 0, color='red', alpha=0.1)
        plt.title('回撤曲线', fontsize=14)
        plt.ylabel('回撤比例', fontsize=12)
        plt.grid(True, alpha=0.3)

        # 情绪得分
        plt.subplot(3, 1, 3)
        sentiment_scores_aligned_plot = self.sentiment_scores.loc[self.portfolio.index]
        sentiment_scores_aligned_plot.plot(label='情绪得分', color='orange', linewidth=1.5)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.title('市场情绪得分 (预测打板收益)', fontsize=14)
        plt.ylabel('情绪得分', fontsize=12)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 月度收益热力图
        plt.figure(figsize=(14, 8))
        monthly_returns = portfolio_normalized.resample('M').last().pct_change()
        monthly_returns = monthly_returns.dropna()
        monthly_returns.index = monthly_returns.index.strftime('%Y-%m')
        monthly_returns = monthly_returns.to_frame('收益')
        monthly_returns['年'] = monthly_returns.index.str[:4]
        monthly_returns['月'] = monthly_returns.index.str[5:]

        pivot_table = monthly_returns.pivot(index='年', columns='月', values='收益')
        month_order = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        pivot_table = pivot_table[month_order]

        sns.heatmap(pivot_table * 100, annot=True, fmt=".1f",
                   cmap="RdYlGn", center=0, linewidths=0.5,
                   annot_kws={"size": 8})
        plt.title('月度收益热力图(%)', fontsize=14)
        plt.xlabel('月份', fontsize=12)
        plt.ylabel('年份', fontsize=12)
        plt.tight_layout()
        plt.show()

# ====================== 主执行流程 ======================
def main():
    logging.info("策略执行开始")

    try:
        with closing(DataLoader()) as data_loader:
            # 1. 数据加载
            print("="*50)
            print("阶段1: 数据加载")
            print("="*50)
            pct_df, open_df, close_df = data_loader.load_data()

            if pct_df is None or pct_df.empty:
                print("数据加载失败，请检查文件路径和格式！")
                return
                
            # 确保PyTorch可用
            try:
                import torch
                print(f"PyTorch version: {torch.__version__}")
                print(f"CUDA available: {torch.cuda.is_available()}")
            except ImportError:
                print("PyTorch未安装，正在安装...")
                import subprocess
                subprocess.check_call(["pip", "install", "torch"])

            # 2. 因子计算
            print("\n" + "="*50)
            print("阶段2: 因子计算")
            print("="*50)
            factor_calculator = FactorCalculator(pct_df, open_df, close_df)
            factors_df = factor_calculator.calculate_factors()

            # 释放内存
            del pct_df, open_df, close_df
            gc.collect()

            # 3. 因子分析
            print("\n" + "="*50)
            print("阶段3: 因子分析")
            print("="*50)
            analyzer = FactorAnalyzer(factors_df)
            ic_results, ic_summary = analyzer.calculate_global_ic()
            print("\n因子IC统计摘要:")
            print(ic_summary)

            # 添加滚动IC分析
            rolling_ic, rolling_ic_summary = analyzer.calculate_rolling_ic(window=252)
            print("\n滚动IC统计摘要:")
            print(rolling_ic_summary)

            analyzer.analyze_factor_relationships()
            analyzer.plot_factor_ts()

            # 4. 构建情绪模型（对比传统模型和DQN模型）
            print("\n" + "="*50)
            print("阶段4: 情绪模型构建")
            print("="*50)
            
            # 传统模型
            print("\n训练传统情绪模型...")
            traditional_model = SentimentModel(factors_df)
            traditional_performance = traditional_model.train_and_compare_models(test_size=0.3)
            
            # DQN模型
            print("\n训练DQN强化学习模型...")
            from sentiment_dqn import DQNSentimentModel
            dqn_model = DQNSentimentModel(factors_df)
            dqn_results = dqn_model.train_model(n_episodes=50, test_size=0.3)
            
            print("\n模型性能对比:")
            # 比较两个模型的性能
            print("传统模型性能:")
            for model, perf in traditional_performance.items():
                print(f"{model}: CV R²={perf['cv_score']:.4f}, Test R²={perf['test_score']:.4f}")
            
            print("\nDQN模型最终性能:")
            final_hist = dqn_results['final_history']
            equity = final_hist['equity']
            ret = equity.pct_change().dropna()
            ann_return = (equity.iloc[-1] / equity.iloc[0]) ** (252 / len(equity)) - 1
            max_dd = (equity / equity.cummax() - 1).min()
            sharpe = (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() > 0 else 0
            print(f"年化收益率: {ann_return:.2%}")
            print(f"最大回撤: {abs(max_dd):.2%}")
            print(f"夏普比率: {sharpe:.2f}")

            print("\n因子权重分配:")
            if traditional_model.weights:
                sorted_weights = sorted(traditional_model.weights.items(), key=lambda item: abs(item[1]), reverse=True)
                for factor, weight in sorted_weights:
                    print(f"{factor}: {weight:.4f}")
            else:
                print("权重未计算，请检查模型训练过程。")

            traditional_model.plot_weights()

            # 5. 创建决策引擎（使用DQN模型的预测）
            print("\n" + "="*50)
            print("阶段5: 决策引擎")
            print("="*50)
            decision_engine = TradingDecisionEngine(factors_df, dqn_model.sentiment_scores)

            if not decision_engine.sentiment_scores_aligned.empty:
                sample_dates = np.random.choice(decision_engine.sentiment_scores_aligned.index,
                                              min(5, len(decision_engine.sentiment_scores_aligned.index)),
                                              replace=False)
                for date in sample_dates:
                    # 持仓股决策
                    holding_decision = decision_engine.make_decision(date, 'holding')
                    print("\n持仓股涨停决策:")
                    for k, v in holding_decision.items():
                        if k == 'indicators':
                            print("市场指标:")
                            for ind, val in v.items():
                                print(f"  {ind}: {val}")
                        else:
                            print(f"{k}: {v}")

                    # 自选股决策
                    watchlist_decision = decision_engine.make_decision(date, 'watchlist')
                    print("\n自选股追涨决策:")
                    for k, v in watchlist_decision.items():
                        if k == 'indicators':
                            continue
                        print(f"{k}: {v}")
            else:
                print("没有足够的有效日期进行决策测试。")

            # 6. 回测验证
            print("\n" + "="*50)
            print("阶段6: 回测验证")
            print("="*50)
            # 分别用传统模型和DQN模型进行回测
            print("\n传统模型回测结果:")
            trad_backtester = StrategyBacktester(factors_df, traditional_model.sentiment_scores)
            trad_portfolio, trad_benchmark = trad_backtester.run_backtest()

            if trad_portfolio is not None and not trad_portfolio.empty:
                trad_performance = trad_backtester.calculate_performance(
                    benchmark_returns=trad_benchmark if not trad_benchmark.empty else None
                )
                if trad_performance:
                    print("\n传统模型策略绩效:")
                    for k, v in trad_performance.items():
                        print(f"{k}: {v:.2f}")
                trad_backtester.plot_results(trad_benchmark)

            print("\nDQN模型回测结果:")
            dqn_backtester = StrategyBacktester(factors_df, dqn_model.sentiment_scores)
            dqn_portfolio, dqn_benchmark = dqn_backtester.run_backtest()

            if dqn_portfolio is not None and not dqn_portfolio.empty:
                dqn_performance = dqn_backtester.calculate_performance(
                    benchmark_returns=dqn_benchmark if not dqn_benchmark.empty else None
                )
                if dqn_performance:
                    print("\nDQN模型策略绩效:")
                    for k, v in dqn_performance.items():
                        print(f"{k}: {v:.2f}")
                dqn_backtester.plot_results(dqn_benchmark)
            else:
                print("回测失败或结果数据为空，无法生成绩效报告和图表。")

            # 7. 保存结果
            print("\n" + "="*50)
            print("阶段7: 结果保存")
            print("="*50)
            # 在当前目录创建results文件夹
            current_dir = os.path.dirname(os.path.abspath(__file__))
            results_dir = os.path.join(current_dir, 'results')
            os.makedirs(results_dir, exist_ok=True)

            try:
                if factors_df is not None and not factors_df.empty:
                    factors_df.to_excel(os.path.join(results_dir, '市场情绪因子.xlsx'))
                # 保存传统模型结果
                if traditional_model.sentiment_scores is not None:
                    traditional_model.sentiment_scores.to_excel(os.path.join(results_dir, '传统模型_情绪得分.xlsx'))
                if trad_portfolio is not None and not trad_portfolio.empty:
                    trad_portfolio.to_excel(os.path.join(results_dir, '传统模型_策略净值.xlsx'))
                if traditional_model.weights is not None:
                    pd.Series(traditional_model.weights).to_excel(os.path.join(results_dir, '传统模型_因子权重.xlsx'))
                if traditional_performance:
                    pd.DataFrame(traditional_performance).T.to_excel(os.path.join(results_dir, '传统模型_性能比较.xlsx'))
                
                # 保存DQN模型结果
                if dqn_model.sentiment_scores is not None:
                    dqn_model.sentiment_scores.to_excel(os.path.join(results_dir, 'DQN模型_情绪得分.xlsx'))
                if dqn_portfolio is not None and not dqn_portfolio.empty:
                    dqn_portfolio.to_excel(os.path.join(results_dir, 'DQN模型_策略净值.xlsx'))
                
                # 保存其他分析结果
                if 'ic_summary' in locals() and ic_summary is not None:
                    ic_summary.to_excel(os.path.join(results_dir, '因子IC分析.xlsx'))
                if 'rolling_ic_summary' in locals() and rolling_ic_summary is not None:
                    rolling_ic_summary.to_excel(os.path.join(results_dir, '滚动IC分析.xlsx'))
                
                # 保存模型对比结果
                if dqn_performance and trad_performance:
                    performance_comparison = pd.DataFrame({
                        '传统模型': trad_performance,
                        'DQN模型': dqn_performance
                    })
                    performance_comparison.to_excel(os.path.join(results_dir, '模型性能对比.xlsx'))

                print(f"\n所有结果已保存到'{results_dir}'文件夹")
            except Exception as e:
                print(f"保存结果失败: {str(e)}")

    except Exception as e:
        logging.error(f"策略执行失败: {str(e)}")
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

        # 尝试保存中间结果
        try:
            if 'factors_df' in locals():
                factors_df.to_pickle(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'factors_df_backup.pkl'))
            if 'dqn_model' in locals() and hasattr(dqn_model, 'sentiment_scores'):
                dqn_model.sentiment_scores.to_pickle(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dqn_scores_backup.pkl'))
            print("中间结果已备份到当前目录")
        except:
            print("无法备份中间结果")

    finally:
        logging.info("策略执行结束")

if __name__ == "__main__":
    main()