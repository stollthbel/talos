import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import deque, namedtuple
import gymnasium as gym
from gymnasium import spaces
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
import ta
from datetime import datetime, timedelta
import asyncio
import websockets
import json
import warnings
import requests
warnings.filterwarnings(‘ignore’)

# ==================== TRANSFORMER BACKBONE ====================

class TransformerEncoder(nn.Module):
“””
Transformer encoder for time-series market data processing
Replaces GRU with self-attention mechanism for temporal memory
“””
def **init**(self, input_dim, d_model=256, nhead=8, num_layers=4, dropout=0.1):
super().**init**()
self.input_dim = input_dim
self.d_model = d_model

```
    # Input projection
    self.input_projection = nn.Linear(input_dim, d_model)
    
    # Positional encoding
    self.pos_encoding = PositionalEncoding(d_model, dropout)
    
    # Transformer layers
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=d_model * 4,
        dropout=dropout,
        activation='gelu',
        batch_first=True
    )
    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
    
    # Layer normalization
    self.layer_norm = nn.LayerNorm(d_model)
    
def forward(self, x, mask=None):
    """
    x: (batch_size, seq_len, input_dim)
    """
    # Project input to d_model
    x = self.input_projection(x)
    
    # Add positional encoding
    x = self.pos_encoding(x)
    
    # Apply transformer
    x = self.transformer(x, src_key_padding_mask=mask)
    
    # Layer norm
    x = self.layer_norm(x)
    
    return x
```

class PositionalEncoding(nn.Module):
def **init**(self, d_model, dropout=0.1, max_len=5000):
super().**init**()
self.dropout = nn.Dropout(p=dropout)

```
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    
    self.register_buffer('pe', pe)
    
def forward(self, x):
    x = x + self.pe[:x.size(0), :]
    return self.dropout(x)
```

# ==================== DEEP Q-LEARNING AGENT ====================

class DuelingDQN(nn.Module):
“””
Dueling Q-Network with Transformer backbone
Separate value and advantage streams for better learning
“””
def **init**(self, input_dim, d_model=256, nhead=8, num_layers=4, action_dim=7):
super().**init**()

```
    # Transformer backbone
    self.transformer = TransformerEncoder(input_dim, d_model, nhead, num_layers)
    
    # Value stream (V(s))
    self.value_stream = nn.Sequential(
        nn.Linear(d_model, 512),
        nn.SiLU(),
        nn.LayerNorm(512),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.SiLU(),
        nn.LayerNorm(256),
        nn.Linear(256, 1)
    )
    
    # Advantage stream (A(s,a))
    self.advantage_stream = nn.Sequential(
        nn.Linear(d_model, 512),
        nn.SiLU(),
        nn.LayerNorm(512),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.SiLU(),
        nn.LayerNorm(256),
        nn.Linear(256, action_dim)
    )
    
def forward(self, x):
    # x shape: (batch_size, seq_len, input_dim)
    
    # Get transformer features
    features = self.transformer(x)
    
    # Use last timestep for Q-values
    last_features = features[:, -1, :]  # (batch_size, d_model)
    
    # Compute value and advantage
    value = self.value_stream(last_features)
    advantage = self.advantage_stream(last_features)
    
    # Combine using dueling architecture
    q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
    
    return q_values
```

class TradingEnvironment(gym.Env):
“””
Custom trading environment for SPX scalping
Actions: [Hold, Buy_Small, Buy_Medium, Buy_Large, Sell_Small, Sell_Medium, Sell_Large]
“””
def **init**(self, data, initial_capital=100000, max_position=10000, transaction_cost=0.001):
super().**init**()

```
    self.data = data
    self.initial_capital = initial_capital
    self.max_position = max_position
    self.transaction_cost = transaction_cost
    
    # Action space: 7 discrete actions
    self.action_space = spaces.Discrete(7)
    
    # Observation space: market features
    self.observation_space = spaces.Box(
        low=-np.inf, high=np.inf,
        shape=(60, data.shape[1]),  # 60 timesteps, n_features
        dtype=np.float32
    )
    
    self.reset()
    
def reset(self, seed=None):
    self.current_step = 60  # Start after lookback period
    self.position = 0
    self.cash = self.initial_capital
    self.portfolio_value = self.initial_capital
    self.trade_count = 0
    self.pnl_history = []
    
    return self._get_observation(), {}

def _get_observation(self):
    start_idx = max(0, self.current_step - 60)
    end_idx = self.current_step
    
    obs = self.data.iloc[start_idx:end_idx].values
    
    # Pad if necessary
    if len(obs) < 60:
        padding = np.zeros((60 - len(obs), obs.shape[1]))
        obs = np.vstack([padding, obs])
        
    return obs.astype(np.float32)

def step(self, action):
    if self.current_step >= len(self.data) - 1:
        return self._get_observation(), 0, True, True, {}
    
    current_price = self.data.iloc[self.current_step]['close']
    
    # Execute action
    reward = self._execute_action(action, current_price)
    
    # Update portfolio value
    self.portfolio_value = self.cash + self.position * current_price
    
    # Move to next step
    self.current_step += 1
    
    # Check if done
    done = (self.current_step >= len(self.data) - 1) or (self.portfolio_value <= 0.5 * self.initial_capital)
    
    return self._get_observation(), reward, done, False, {
        'portfolio_value': self.portfolio_value,
        'position': self.position,
        'cash': self.cash,
        'trade_count': self.trade_count
    }

def _execute_action(self, action, current_price):
    """
    Execute trading action and return reward
    Actions: [Hold, Buy_Small, Buy_Medium, Buy_Large, Sell_Small, Sell_Medium, Sell_Large]
    """
    action_map = {
        0: 0,      # Hold
        1: 1000,   # Buy Small
        2: 2500,   # Buy Medium
        3: 5000,   # Buy Large
        4: -1000,  # Sell Small
        5: -2500,  # Sell Medium
        6: -5000   # Sell Large
    }
    
    trade_size = action_map[action]
    
    # Calculate position change
    new_position = self.position + trade_size
    
    # Position limits
    new_position = np.clip(new_position, -self.max_position, self.max_position)
    actual_trade = new_position - self.position
    
    # Execute trade
    if actual_trade != 0:
        trade_value = actual_trade * current_price
        transaction_cost = abs(trade_value) * self.transaction_cost
        
        # Check if we have enough cash
        if self.cash >= trade_value + transaction_cost:
            self.cash -= trade_value + transaction_cost
            self.position = new_position
            self.trade_count += 1
            
            # Calculate reward
            reward = self._calculate_reward(action, actual_trade, current_price)
        else:
            reward = -10  # Penalty for insufficient cash
    else:
        reward = self._calculate_reward(action, 0, current_price)
    
    return reward

def _calculate_reward(self, action, trade_size, current_price):
    """
    Trident-inspired reward function with risk-aware shaping
    """
    # Base reward: portfolio value change
    if len(self.pnl_history) > 0:
        pnl_change = self.portfolio_value - self.pnl_history[-1]
        base_reward = pnl_change / self.initial_capital * 1000  # Scale reward
    else:
        base_reward = 0
    
    # Risk penalty: position size
    position_ratio = abs(self.position) / self.max_position
    risk_penalty = -position_ratio * 2
    
    # Overtrading penalty
    if self.trade_count > 100:  # Limit trades per episode
        overtrading_penalty = -5
    else:
        overtrading_penalty = 0
    
    # Action-specific rewards
    if action == 0:  # Hold
        action_reward = 0.1  # Small reward for patience
    else:
        action_reward = 0
    
    total_reward = base_reward + risk_penalty + overtrading_penalty + action_reward
    
    # Update history
    self.pnl_history.append(self.portfolio_value)
    
    return total_reward
```

# ==================== FEATURE ENGINEERING ====================

class MarketFeatureExtractor:
“””
Extract comprehensive market features for transformer input
“””
def **init**(self):
self.scaler = StandardScaler()
self.fitted = False

```
def extract_features(self, data):
    """
    Extract features from OHLCV data
    """
    df = data.copy()


    # Price/volume features
    #features we have - price, volume, smoothed regressions, options vol surface? skew score, time pct, vol state

    # OHLCV ratios
    df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
    df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    df['hl2'] = (df['high'] + df['low']) / 2
    df['oc2'] = (df['open'] + df['close']) / 2
    df['c1_o'] = df['close'] / df['open']
    df['c1_h'] = df['close'] / df['high']
    df['c1_l'] = df['close'] / df['low']
    df['o1_c'] = df['open'] / df['close']
    df['h1_o'] = df['high'] / df['open']
    df['l1_o'] = df['low'] / df['open']

    # Rolling statistics
    for window in [5, 10, 20, 50, 100, 200]:
        df[f'close_ma_{window}'] = df['close'].rolling(window).mean()
        df[f'close_std_{window}'] = df['close'].rolling(window).std()
        df[f'volume_ma_{window}'] = df['volume'].rolling(window).mean()
        df[f'volume_std_{window}'] = df['volume'].rolling(window).std()
        df[f'returns_ma_{window}'] = df['returns'].rolling(window).mean()
        df[f'returns_std_{window}'] = df['returns'].rolling(window).std()
        df[f'high_ma_{window}'] = df['high'].rolling(window).mean()
        df[f'low_ma_{window}'] = df['low'].rolling(window).mean()

    # Momentum features
    df['momentum_10'] = df['close'] - df['close'].shift(10)
    df['momentum_20'] = df['close'] - df['close'].shift(20)
    df['momentum_50'] = df['close'] - df['close'].shift(50)

    # Cumulative returns
    df['cum_return_5'] = df['returns'].rolling(5).sum()
    df['cum_return_20'] = df['returns'].rolling(20).sum()
    df['cum_return_60'] = df['returns'].rolling(60).sum()

    # Drawdown
    df['rolling_max'] = df['close'].rolling(20).max()
    df['drawdown'] = (df['close'] - df['rolling_max']) / df['rolling_max']

    # Price change features
    df['close_diff_1'] = df['close'].diff(1)
    df['close_diff_5'] = df['close'].diff(5)
    df['close_diff_10'] = df['close'].diff(10)
    df['high_diff_1'] = df['high'].diff(1)
    df['low_diff_1'] = df['low'].diff(1)

    # Volatility features
    df['realized_vol_5'] = df['returns'].rolling(5).std() * np.sqrt(252)
    df['realized_vol_20'] = df['returns'].rolling(20).std() * np.sqrt(252)
    df['vol_change_5'] = df['realized_vol_5'].diff(5)
    df['vol_zscore_20'] = (df['realized_vol_20'] - df['realized_vol_20'].rolling(100).mean()) / df['realized_vol_20'].rolling(100).std()

    # Price percentiles
    df['close_pct_20'] = df['close'].rolling(20).apply(lambda x: (x[-1] - np.min(x)) / (np.max(x) - np.min(x)) if np.max(x) != np.min(x) else 0)
    df['volume_pct_20'] = df['volume'].rolling(20).apply(lambda x: (x[-1] - np.min(x)) / (np.max(x) - np.min(x)) if np.max(x) != np.min(x) else 0)

    # Candle features
    df['candle_body'] = abs(df['close'] - df['open'])
    df['candle_range'] = df['high'] - df['low']
    df['upper_shadow'] = df['high'] - np.maximum(df['close'], df['open'])
    df['lower_shadow'] = np.minimum(df['close'], df['open']) - df['low']
    df['body_to_range'] = df['candle_body'] / (df['candle_range'] + 1e-6)

    # Gaps
    df['gap'] = df['open'] - df['close'].shift(1)
    df['gap_pct'] = df['gap'] / df['close'].shift(1)

    # Rolling correlations
    df['corr_close_volume_20'] = df['close'].rolling(20).corr(df['volume'])
    df['corr_high_low_20'] = df['high'].rolling(20).corr(df['low'])

    # Rolling skew/kurtosis
    df['skew_20'] = df['returns'].rolling(20).skew()
    df['kurt_20'] = df['returns'].rolling(20).kurt()

    # Price acceleration
    df['accel_5'] = df['close_diff_1'].rolling(5).mean().diff()
    df['accel_20'] = df['close_diff_1'].rolling(20).mean().diff()

    # Turnover (volume relative to rolling mean)
    df['turnover_20'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-6)

    # Intraday features (if intraday data)
    if hasattr(df.index, 'hour'):
        df['is_open'] = ((df.index.hour >= 9) & (df.index.hour <= 16)).astype(int)
        df['minute'] = df.index.minute if hasattr(df.index, 'minute') else 0
        df['time_of_day'] = df.index.hour * 60 + df['minute']
        df['time_pct'] = df['time_of_day'] / (16 * 60)

    # Calendar features
    df['month'] = df.index.month if hasattr(df.index, 'month') else 0
    df['quarter'] = df.index.quarter if hasattr(df.index, 'quarter') else 0
    df['year'] = df.index.year if hasattr(df.index, 'year') else 0
    df['is_month_start'] = df.index.is_month_start.astype(int) if hasattr(df.index, 'is_month_start') else 0
    df['is_month_end'] = df.index.is_month_end.astype(int) if hasattr(df.index, 'is_month_end') else 0

    # Rolling min/max
    df['rolling_min_20'] = df['close'].rolling(20).min()
    df['rolling_max_20'] = df['close'].rolling(20).max()
    df['range_20'] = df['rolling_max_20'] - df['rolling_min_20']

    # ATR normalized
    df['atr_norm'] = df['atr'] / (df['close'].rolling(20).mean() + 1e-6)

    # MACD cross
    df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int)

    # RSI overbought/oversold
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)

    # Bollinger band position
    df['bb_pos'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-6)

    # Slope of moving averages
    for window in [5, 10, 20]:
        df[f'slope_ma_{window}'] = df[f'close_ma_{window}'].diff()

    # Crosses
    df['ma5_above_ma20'] = (df['close_ma_5'] > df['close_ma_20']).astype(int)
    df['ma10_above_ma50'] = (df['close_ma_10'] > df['close_ma_50']).astype(int)

    # Price relative to moving averages
    for window in [5, 10, 20, 50, 100, 200]:
        df[f'close_above_ma_{window}'] = (df['close'] > df[f'close_ma_{window}']).astype(int)

    # Volume spike
    df['volume_spike'] = (df['volume'] > 2 * df['volume_ma_20']).astype(int)

    # Rolling z-score of close
    df['close_zscore_20'] = (df['close'] - df['close_ma_20']) / (df['close_std_20'] + 1e-6)

    # Rolling Sharpe ratio
    df['sharpe_20'] = df['returns_ma_20'] / (df['returns_std_20'] + 1e-6)

    # Rolling max drawdown
    roll_max = df['close'].rolling(20, min_periods=1).max()
    daily_drawdown = df['close'] / roll_max - 1.0
    df['max_drawdown_20'] = daily_drawdown.rolling(20, min_periods=1).min()

    # Optionally: add more features from options data, news sentiment, ETF flows, etc.
    
    # === News Sentiment Features ===
    # Assume you have a DataFrame `news_sentiment` indexed by datetime with a 'sentiment_score' column
    # Merge news sentiment to price data by timestamp (forward fill to align)
    if hasattr(self, 'news_sentiment') and self.news_sentiment is not None:
        df = df.merge(self.news_sentiment[['sentiment_score']], left_index=True, right_index=True, how='left')
        df['news_sentiment'] = df['sentiment_score'].fillna(method='ffill').fillna(0)
        df.drop(columns=['sentiment_score'], inplace=True)
    else:
        df['news_sentiment'] = 0.0

    # === Trump Tweet Sentiment Features (LLM-based) ===
    # Example: Use an LLM API to analyze Trump's tweets from X (Twitter)
    # Assume you have a DataFrame `trump_tweets` indexed by datetime with a 'tweet_text' column
    # You can call an LLM API to get sentiment for each tweet and cache the results

    if hasattr(self, 'trump_tweets') and self.trump_tweets is not None:
        # If 'tweet_sentiment' not present, compute it using LLM API
        if 'tweet_sentiment' not in self.trump_tweets.columns:

            def get_llm_sentiment(text):
                # Replace with your LLM API endpoint and authentication
                api_url = "https://api.your-llm-provider.com/v1/sentiment"
                payload = {"text": text}
                headers = {"Authorization": "Bearer YOUR_API_KEY"}
                try:
                    response = requests.post(api_url, json=payload, headers=headers, timeout=5)
                    if response.status_code == 200:
                        return response.json().get("sentiment_score", 0)
                    else:
                        return 0
                except Exception:
                    return 0

            self.trump_tweets['tweet_sentiment'] = self.trump_tweets['tweet_text'].apply(get_llm_sentiment)

        # Merge tweet sentiment to price data by timestamp (forward fill to align)
        df = df.merge(self.trump_tweets[['tweet_sentiment']], left_index=True, right_index=True, how='left')
        df['trump_tweet_sentiment'] = df['tweet_sentiment'].fillna(method='ffill').fillna(0)
        df.drop(columns=['tweet_sentiment'], inplace=True)
    else:
        df['trump_tweet_sentiment'] = 0.0
    # Price features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']
    
    # Volume features
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # Technical indicators
    df['sma_5'] = ta.trend.sma_indicator(df['close'], 5)
    df['sma_10'] = ta.trend.sma_indicator(df['close'], 10)
    df['sma_20'] = ta.trend.sma_indicator(df['close'], 20)
    df['ema_12'] = ta.trend.ema_indicator(df['close'], 12)
    df['ema_26'] = ta.trend.ema_indicator(df['close'], 26)
    
    # MACD
    df['macd'] = ta.trend.macd(df['close'])
    df['macd_signal'] = ta.trend.macd_signal(df['close'])
    df['macd_histogram'] = ta.trend.macd_diff(df['close'])
    
    # RSI
    df['rsi'] = ta.momentum.rsi(df['close'])
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # Stochastic
    df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
    df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
    
    # Williams %R
    df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
    
    # Average True Range
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
    
    # Commodity Channel Index
    df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
    
    # Money Flow Index
    df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])
    
    # On Balance Volume
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    
    # Volatility features
    df['volatility'] = df['returns'].rolling(20).std()
    df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(60).mean()
    
    # Market microstructure
    df['spread'] = df['high'] - df['low']
    df['midpoint'] = (df['high'] + df['low']) / 2
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # Time-based features
    df['hour'] = df.index.hour if hasattr(df.index, 'hour') else 0
    df['day_of_week'] = df.index.dayofweek if hasattr(df.index, 'dayofweek') else 0
    
    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
    
    # Forward fill and drop NaN
    df = df.fillna(method='ffill').dropna()
    
    # Select feature columns
    feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
    
    # Normalize features
    if not self.fitted:
        self.scaler.fit(df[feature_cols])
        self.fitted = True
    
    df[feature_cols] = self.scaler.transform(df[feature_cols])
    
    return df[feature_cols]
```

# ==================== CHRIS HAHN PORTFOLIO INTEGRATION ====================

class HahnPortfolioManager:
“””
Manage Chris Hahn-style portfolio with algorithmic execution
“””
def **init**(self, initial_capital=1000000):
self.initial_capital = initial_capital
self.hahn_universe = {
‘MSFT’: {‘target_weight’: 0.15, ‘roe’: 35.2, ‘roa’: 15.1},
‘V’: {‘target_weight’: 0.13, ‘roe’: 35.8, ‘roa’: 15.3},
‘MCO’: {‘target_weight’: 0.14, ‘roe’: 25.4, ‘roa’: 10.2},
‘SPGI’: {‘target_weight’: 0.12, ‘roe’: 30.1, ‘roa’: 12.3},
‘GE’: {‘target_weight’: 0.22, ‘roe’: 20.2, ‘roa’: 8.1},
‘CNI’: {‘target_weight’: 0.08, ‘roe’: 19.8, ‘roa’: 7.9},
‘GOOGL’: {‘target_weight’: 0.05, ‘roe’: 25.7, ‘roa’: 12.1},
‘RACE’: {‘target_weight’: 0.05, ‘roe’: 29.8, ‘roa’: 14.9},
‘ABNB’: {‘target_weight’: 0.03, ‘roe’: 22.1, ‘roa’: 9.4},
‘SPY’: {‘target_weight’: 0.03, ‘roe’: 20.0, ‘roa’: 12.0}  # SPX scalping
}

```
    self.current_positions = {}
    self.scalping_agents = {}
    
def initialize_scalping_agents(self):
    """Initialize DQN agents for each symbol"""
    for symbol in self.hahn_universe.keys():
        # Get market data
        data = self.fetch_market_data(symbol)
        
        if data is not None:
            # Extract features
            extractor = MarketFeatureExtractor()
            features = extractor.extract_features(data)
            
            # Create trading environment
            env = TradingEnvironment(features, 
                                   initial_capital=self.initial_capital * self.hahn_universe[symbol]['target_weight'])
            
            # Create DQN agent
            agent = DQNAgent(
                state_dim=features.shape[1],
                action_dim=7,
                lr=0.001,
                gamma=0.99,
                epsilon=0.1
            )
            
            self.scalping_agents[symbol] = {
                'agent': agent,
                'env': env,
                'extractor': extractor,
                'data': data
            }

def fetch_market_data(self, symbol, period="1y"):
    """Fetch market data for symbol"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval="1h")  # Hourly data for scalping
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None
```

# ==================== DQN AGENT ====================

class DQNAgent:
“””
Deep Q-Network agent with experience replay and target network
“””
def **init**(self, state_dim, action_dim=7, lr=0.001, gamma=0.99, epsilon=0.1):
self.state_dim = state_dim
self.action_dim = action_dim
self.lr = lr
self.gamma = gamma
self.epsilon = epsilon

```
    # Networks
    self.q_network = DuelingDQN(state_dim, action_dim=action_dim)
    self.target_network = DuelingDQN(state_dim, action_dim=action_dim)
    
    # Copy weights to target network
    self.target_network.load_state_dict(self.q_network.state_dict())
    
    # Optimizer
    self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
    
    # Experience replay
    self.replay_buffer = ReplayBuffer(10000)
    
    # Training parameters
    self.batch_size = 32
    self.update_frequency = 4
    self.target_update_frequency = 1000
    self.step_count = 0
    
def select_action(self, state, training=True):
    """Select action using epsilon-greedy policy"""
    if training and np.random.random() < self.epsilon:
        return np.random.randint(self.action_dim)
    
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

def store_experience(self, state, action, reward, next_state, done):
    """Store experience in replay buffer"""
    self.replay_buffer.add(state, action, reward, next_state, done)

def train(self):
    """Train the DQN"""
    if len(self.replay_buffer) < self.batch_size:
        return
    
    # Sample batch
    batch = self.replay_buffer.sample(self.batch_size)
    states, actions, rewards, next_states, dones = batch
    
    # Convert to tensors
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.BoolTensor(dones)
    
    # Current Q values
    current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
    
    # Next Q values from target network
    with torch.no_grad():
        next_q_values = self.target_network(next_states).max(1)[0]
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
    
    # Loss
    loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
    
    # Optimize
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
    self.optimizer.step()
    
    # Update target network
    self.step_count += 1
    if self.step_count % self.target_update_frequency == 0:
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    # Decay epsilon
    self.epsilon = max(0.01, self.epsilon * 0.995)
    
    return loss.item()
```

# ==================== REPLAY BUFFER ====================

Experience = namedtuple(‘Experience’, [‘state’, ‘action’, ‘reward’, ‘next_state’, ‘done’])

class ReplayBuffer:
“”“Experience replay buffer for DQN”””
def **init**(self, capacity):
self.capacity = capacity
self.buffer = deque(maxlen=capacity)

```
def add(self, state, action, reward, next_state, done):
    experience = Experience(state, action, reward, next_state, done)
    self.buffer.append(experience)

def sample(self, batch_size):
    indices = np.random.choice(len(self.buffer), batch_size, replace=False)
    batch = [self.buffer[i] for i in indices]
    
    states = np.array([e.state for e in batch])
    actions = np.array([e.action for e in batch])
    rewards = np.array([e.reward for e in batch])
    next_states = np.array([e.next_state for e in batch])
    dones = np.array([e.done for e in batch])
    
    return states, actions, rewards, next_states, dones

def __len__(self):
    return len(self.buffer)
```

# ==================== MAIN TRIDENT SYSTEM ====================

class TridentTradingSystem:
“””
Complete Trident trading system integrating:
- Chris Hahn portfolio selection
- Transformer-based market analysis
- Deep Q-learning for execution
- Risk management and capital scaling
“””
def **init**(self, initial_capital=1000000):
self.initial_capital = initial_capital
self.portfolio_manager = HahnPortfolioManager(initial_capital)
self.performance_tracker = PerformanceTracker()

```
    # Capital scaling phases
    self.capital_phases = {
        'phase_1': {'range': (30000, 100000), 'strategy': 'conservative'},
        'phase_2': {'range': (100000, 500000), 'strategy': 'aggressive'},
        'phase_3': {'range': (500000, 2000000), 'strategy': 'multi_strike'},
        'phase_4': {'range': (2000000, 8000000), 'strategy': 'synthetic_vol'}
    }
    
    self.current_phase = self.determine_phase()
    
def determine_phase(self):
    """Determine current capital phase"""
    for phase, config in self.capital_phases.items():
        if config['range'][0] <= self.initial_capital <= config['range'][1]:
            return phase
    return 'phase_4'  # Default to highest phase

def initialize_system(self):
    """Initialize complete trading system"""
    print("🚀 Initializing Trident Trading System...")
    
    # Initialize portfolio manager
    self.portfolio_manager.initialize_scalping_agents()
    
    # Initialize performance tracking
    self.performance_tracker.initialize()
    
    print(f"✅ System initialized with ${self.initial_capital:,.2f}")
    print(f"📊 Current phase: {self.current_phase}")
    print(f"🎯 Active symbols: {list(self.portfolio_manager.hahn_universe.keys())}")
    
def run_trading_session(self, duration_hours=24):
    """Run complete trading session"""
    print(f"\n🏃 Starting {duration_hours}-hour trading session...")
    
    session_results = {}
    
    for symbol, agent_config in self.portfolio_manager.scalping_agents.items():
        print(f"\n📈 Trading {symbol}...")
        
        # Run DQN training/trading
        results = self.run_symbol_trading(symbol, agent_config, duration_hours)
        session_results[symbol] = results
        
        # Update performance tracking
        self.performance_tracker.update_symbol_performance(symbol, results)
    
    # Generate session report
    self.generate_session_report(session_results)
    
    return session_results

def run_symbol_trading(self, symbol, agent_config, duration_hours):
    """Run trading for specific symbol"""
    agent = agent_config['agent']
    env = agent_config['env']
    
    # Training phase
    print(f"🧠 Training {symbol} agent...")
    training_results = self.train_agent(agent, env, episodes=100)
    
    # Live trading simulation
    print(f"💹 Live trading {symbol}...")
    trading_results = self.simulate_live_trading(agent, env, duration_hours)
    
    return {
        'training': training_results,
        'trading': trading_results,
        'symbol': symbol
    }

def train_agent(self, agent, env, episodes=100):
    """Train DQN agent"""
    episode_rewards = []
    losses = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_losses = []
        
        while True:
            action = agent.select_action(state, training=True)
            next_state, reward, done, truncated, info = env.step(action)
            
            agent.store_experience(state, action, reward, next_state, done)
            
            if agent.step_count % agent.update_frequency == 0:
                loss = agent.train()
                if loss is not None:
                    episode_losses.append(loss)
            
            state = next_state
            episode_reward += reward
            
            if done or truncated:
                break
        
        episode_rewards.append(episode_reward)
        if episode_losses:
            losses.append(np.mean(episode_losses))
        
        if episode % 20 == 0:
            avg_reward = np.mean(episode_rewards[-20:])
            print(f"    Episode {episode}, Avg Reward: {avg_reward:.2f}")
    
    return {
        'episode_rewards': episode_rewards,
        'losses': losses,
        'final_epsilon': agent.epsilon
    }

def simulate_live_trading(self, agent, env, duration_hours):
    """Simulate live trading session"""
    state, _ = env.reset()
    total_reward = 0
    actions_taken = []
    portfolio_values = []
    
    steps = min(duration_hours * 4, len(env.data) - env.current_step)  # 15-min intervals
    
    for step in range(steps):
        action = agent.select_action(state, training=False)
        next_state, reward, done, truncated, info = env.step(action)
        
        actions_taken.append(action)
        portfolio_values.append(info['portfolio_value'])
        total_reward += reward
        
        state = next_state
        
        if done or truncated:
            break
    
    return {
        'total_reward': total_reward,
        'actions_taken': actions_taken,
        'portfolio_values': portfolio_values,
        'final_portfolio_value': portfolio_values[-1] if portfolio_values else env.initial_capital,
        'total_trades': info.get('trade_count', 0)
    }

def generate_session_report(self, session_results):
    """Generate comprehensive session report"""
    print("\n" + "="*60)
    print("📊 TRIDENT TRADING SESSION REPORT")
    print("="*60)
    
    total_pnl = 0
    total_trades = 0
    
    for symbol, results in session_results.items():
        trading_results = results['trading']
        initial_value = self.initial_capital * self.portfolio_manager.hahn_universe[symbol]['target_weight']
        final_value = trading_results['final_portfolio_value']
        pnl = final_value - initial_value
        pnl_pct = (pnl / initial_value) * 100
        
        total_pnl += pnl
        total_trades += trading_results['total_trades']
        
        print(f"\n{symbol}:")
        print(f"  Initial: ${initial_value:,.2f}")
        print(f"
```