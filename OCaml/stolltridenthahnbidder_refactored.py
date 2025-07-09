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
warnings.filterwarnings('ignore')

# ==================== CONSTANTS ====================

class TradingConstants:
    """Centralized constants for trading system"""
    
    # Neural Network Architecture
    TRANSFORMER_D_MODEL = 256
    TRANSFORMER_NHEAD = 8
    TRANSFORMER_NUM_LAYERS = 4
    TRANSFORMER_DROPOUT = 0.1
    TRANSFORMER_FEEDFORWARD_MULTIPLIER = 4
    
    # Trading Environment
    INITIAL_CAPITAL = 100000
    MAX_POSITION = 10000
    TRANSACTION_COST = 0.001
    LOOKBACK_PERIOD = 60
    ACTION_SPACE_SIZE = 7
    MAX_TRADES_PER_EPISODE = 100
    
    # Action Mappings
    ACTION_HOLD = 0
    ACTION_BUY_SMALL = 1
    ACTION_BUY_MEDIUM = 2
    ACTION_BUY_LARGE = 3
    ACTION_SELL_SMALL = 4
    ACTION_SELL_MEDIUM = 5
    ACTION_SELL_LARGE = 6
    
    # Trade Sizes
    TRADE_SIZE_SMALL = 1000
    TRADE_SIZE_MEDIUM = 2500
    TRADE_SIZE_LARGE = 5000
    
    # DQN Parameters
    DQN_LEARNING_RATE = 0.001
    DQN_GAMMA = 0.99
    DQN_EPSILON = 0.1
    DQN_MIN_EPSILON = 0.01
    DQN_EPSILON_DECAY = 0.995
    DQN_BATCH_SIZE = 32
    DQN_UPDATE_FREQUENCY = 4
    DQN_TARGET_UPDATE_FREQUENCY = 1000
    DQN_REPLAY_BUFFER_SIZE = 10000
    DQN_GRADIENT_CLIP = 1.0
    
    # Network Layer Sizes
    VALUE_STREAM_HIDDEN_1 = 512
    VALUE_STREAM_HIDDEN_2 = 256
    ADVANTAGE_STREAM_HIDDEN_1 = 512
    ADVANTAGE_STREAM_HIDDEN_2 = 256
    DROPOUT_RATE = 0.2
    
    # Technical Indicators
    ROLLING_WINDOWS = [5, 10, 20, 50, 100, 200]
    MOMENTUM_PERIODS = [10, 20, 50]
    CUMULATIVE_RETURN_PERIODS = [5, 20, 60]
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    RSI_PERIOD = 14
    
    # MACD Parameters
    MACD_FAST_PERIOD = 12
    MACD_SLOW_PERIOD = 26
    MACD_SIGNAL_PERIOD = 9
    
    # Bollinger Bands
    BB_PERIOD = 20
    BB_STD_DEV = 2
    
    # Volatility Calculations
    VOLATILITY_PERIODS = [5, 20]
    VOLATILITY_ANNUALIZATION = 252
    
    # Correlation and Statistics
    CORRELATION_PERIODS = [20]
    SKEW_KURTOSIS_PERIODS = [20]
    
    # Time Features
    MARKET_OPEN_HOUR = 9
    MARKET_CLOSE_HOUR = 16
    MINUTES_PER_HOUR = 60
    
    # Position Limits
    POSITION_RATIO_PENALTY = 2
    INSUFFICIENT_CASH_PENALTY = -10
    HOLD_REWARD = 0.1
    OVERTRADING_PENALTY = -5
    
    # Portfolio Management
    MAX_DRAWDOWN_THRESHOLD = 0.5
    PORTFOLIO_VALUE_SCALE = 1000
    
    # Feature Engineering
    PRICE_PERCENTILE_PERIODS = [20]
    VOLUME_SPIKE_MULTIPLIER = 2
    Z_SCORE_PERIODS = [20, 100]
    
    # Portfolio Phase Thresholds
    PHASE_1_MIN = 30000
    PHASE_1_MAX = 100000
    PHASE_2_MIN = 100000
    PHASE_2_MAX = 500000
    PHASE_3_MIN = 500000
    PHASE_3_MAX = 2000000
    PHASE_4_MIN = 2000000
    PHASE_4_MAX = 8000000
    
    # Training Parameters
    TRAINING_EPISODES = 100
    TRAINING_PROGRESS_INTERVAL = 20
    
    # Data Fetching
    DEFAULT_PERIOD = "1y"
    DEFAULT_INTERVAL = "1h"
    
    # API and External Data
    API_TIMEOUT = 5
    SENTIMENT_FILL_VALUE = 0.0
    
    # Lag Features
    LAG_PERIODS = [1, 2, 3, 5, 10]
    
    # Market Microstructure
    EPSILON_SMALL = 1e-6
    
    # Positional Encoding
    POSITIONAL_ENCODING_MAX_LEN = 5000
    POSITIONAL_ENCODING_DROPOUT = 0.1
    POSITIONAL_ENCODING_BASE = 10000.0


# ==================== HAHN PORTFOLIO CONSTANTS ====================

class HahnPortfolioConstants:
    """Chris Hahn portfolio configuration"""
    
    DEFAULT_INITIAL_CAPITAL = 1000000
    
    PORTFOLIO_UNIVERSE = {
        'MSFT': {'target_weight': 0.15, 'roe': 35.2, 'roa': 15.1},
        'V': {'target_weight': 0.13, 'roe': 35.8, 'roa': 15.3},
        'MCO': {'target_weight': 0.14, 'roe': 25.4, 'roa': 10.2},
        'SPGI': {'target_weight': 0.12, 'roe': 30.1, 'roa': 12.3},
        'GE': {'target_weight': 0.22, 'roe': 20.2, 'roa': 8.1},
        'CNI': {'target_weight': 0.08, 'roe': 19.8, 'roa': 7.9},
        'GOOGL': {'target_weight': 0.05, 'roe': 25.7, 'roa': 12.1},
        'RACE': {'target_weight': 0.05, 'roe': 29.8, 'roa': 14.9},
        'ABNB': {'target_weight': 0.03, 'roe': 22.1, 'roa': 9.4},
        'SPY': {'target_weight': 0.03, 'roe': 20.0, 'roa': 12.0}
    }
    
    CAPITAL_PHASES = {
        'phase_1': {'range': (30000, 100000), 'strategy': 'conservative'},
        'phase_2': {'range': (100000, 500000), 'strategy': 'aggressive'},
        'phase_3': {'range': (500000, 2000000), 'strategy': 'multi_strike'},
        'phase_4': {'range': (2000000, 8000000), 'strategy': 'synthetic_vol'}
    }


# ==================== TRANSFORMER BACKBONE ====================

class TransformerEncoder(nn.Module):
    """
    Transformer encoder for time-series market data processing
    Replaces GRU with self-attention mechanism for temporal memory
    """
    def __init__(self, input_dim, 
                 d_model=TradingConstants.TRANSFORMER_D_MODEL, 
                 nhead=TradingConstants.TRANSFORMER_NHEAD, 
                 num_layers=TradingConstants.TRANSFORMER_NUM_LAYERS, 
                 dropout=TradingConstants.TRANSFORMER_DROPOUT):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * TradingConstants.TRANSFORMER_FEEDFORWARD_MULTIPLIER,
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, 
                 dropout=TradingConstants.POSITIONAL_ENCODING_DROPOUT, 
                 max_len=TradingConstants.POSITIONAL_ENCODING_MAX_LEN):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(TradingConstants.POSITIONAL_ENCODING_BASE) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# ==================== DEEP Q-LEARNING AGENT ====================

class DuelingDQN(nn.Module):
    """
    Dueling Q-Network with Transformer backbone
    Separate value and advantage streams for better learning
    """
    def __init__(self, input_dim, 
                 d_model=TradingConstants.TRANSFORMER_D_MODEL, 
                 nhead=TradingConstants.TRANSFORMER_NHEAD, 
                 num_layers=TradingConstants.TRANSFORMER_NUM_LAYERS, 
                 action_dim=TradingConstants.ACTION_SPACE_SIZE):
        super().__init__()
        
        # Transformer backbone
        self.transformer = TransformerEncoder(input_dim, d_model, nhead, num_layers)
        
        # Value stream (V(s))
        self.value_stream = nn.Sequential(
            nn.Linear(d_model, TradingConstants.VALUE_STREAM_HIDDEN_1),
            nn.SiLU(),
            nn.LayerNorm(TradingConstants.VALUE_STREAM_HIDDEN_1),
            nn.Dropout(TradingConstants.DROPOUT_RATE),
            nn.Linear(TradingConstants.VALUE_STREAM_HIDDEN_1, TradingConstants.VALUE_STREAM_HIDDEN_2),
            nn.SiLU(),
            nn.LayerNorm(TradingConstants.VALUE_STREAM_HIDDEN_2),
            nn.Linear(TradingConstants.VALUE_STREAM_HIDDEN_2, 1)
        )
        
        # Advantage stream (A(s,a))
        self.advantage_stream = nn.Sequential(
            nn.Linear(d_model, TradingConstants.ADVANTAGE_STREAM_HIDDEN_1),
            nn.SiLU(),
            nn.LayerNorm(TradingConstants.ADVANTAGE_STREAM_HIDDEN_1),
            nn.Dropout(TradingConstants.DROPOUT_RATE),
            nn.Linear(TradingConstants.ADVANTAGE_STREAM_HIDDEN_1, TradingConstants.ADVANTAGE_STREAM_HIDDEN_2),
            nn.SiLU(),
            nn.LayerNorm(TradingConstants.ADVANTAGE_STREAM_HIDDEN_2),
            nn.Linear(TradingConstants.ADVANTAGE_STREAM_HIDDEN_2, action_dim)
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


class TradingEnvironment(gym.Env):
    """
    Reinforcement learning in a trading environment
    Actions: [Hold, Buy_Small, Buy_Medium, Buy_Large, Sell_Small, Sell_Medium, Sell_Large]
    """
    def __init__(self, data, 
                 initial_capital=TradingConstants.INITIAL_CAPITAL,
                 max_position=TradingConstants.MAX_POSITION,
                 transaction_cost=TradingConstants.TRANSACTION_COST):
        super().__init__()
        
        self.data = data
        self.initial_capital = initial_capital
        self.max_position = max_position
        self.transaction_cost = transaction_cost
        
        # Action space: 7 discrete actions
        self.action_space = spaces.Discrete(TradingConstants.ACTION_SPACE_SIZE)
        
        # Observation space: market features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(TradingConstants.LOOKBACK_PERIOD, data.shape[1]),
            dtype=np.float32
        )
        
        # Action mapping
        self.action_map = {
            TradingConstants.ACTION_HOLD: 0,
            TradingConstants.ACTION_BUY_SMALL: TradingConstants.TRADE_SIZE_SMALL,
            TradingConstants.ACTION_BUY_MEDIUM: TradingConstants.TRADE_SIZE_MEDIUM,
            TradingConstants.ACTION_BUY_LARGE: TradingConstants.TRADE_SIZE_LARGE,
            TradingConstants.ACTION_SELL_SMALL: -TradingConstants.TRADE_SIZE_SMALL,
            TradingConstants.ACTION_SELL_MEDIUM: -TradingConstants.TRADE_SIZE_MEDIUM,
            TradingConstants.ACTION_SELL_LARGE: -TradingConstants.TRADE_SIZE_LARGE
        }
        
        self.reset()
        
    def reset(self, seed=None):
        self.current_step = TradingConstants.LOOKBACK_PERIOD
        self.position = 0
        self.cash = self.initial_capital
        self.portfolio_value = self.initial_capital
        self.trade_count = 0
        self.pnl_history = []
        
        return self._get_observation(), {}

    def _get_observation(self):
        start_idx = max(0, self.current_step - TradingConstants.LOOKBACK_PERIOD)
        end_idx = self.current_step
        
        obs = self.data.iloc[start_idx:end_idx].values
        
        # Pad if necessary
        if len(obs) < TradingConstants.LOOKBACK_PERIOD:
            padding = np.zeros((TradingConstants.LOOKBACK_PERIOD - len(obs), obs.shape[1]))
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
        done = (self.current_step >= len(self.data) - 1) or \
               (self.portfolio_value <= TradingConstants.MAX_DRAWDOWN_THRESHOLD * self.initial_capital)
        
        return self._get_observation(), reward, done, False, {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'cash': self.cash,
            'trade_count': self.trade_count
        }

    def _execute_action(self, action, current_price):
        """Execute trading action and return reward"""
        trade_size = self.action_map[action]
        
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
                reward = TradingConstants.INSUFFICIENT_CASH_PENALTY
        else:
            reward = self._calculate_reward(action, 0, current_price)
        
        return reward

    def _calculate_reward(self, action, trade_size, current_price):
        """Trident-inspired reward function with risk-aware shaping"""
        # Base reward: portfolio value change
        if len(self.pnl_history) > 0:
            pnl_change = self.portfolio_value - self.pnl_history[-1]
            base_reward = pnl_change / self.initial_capital * TradingConstants.PORTFOLIO_VALUE_SCALE
        else:
            base_reward = 0
        
        # Risk penalty: position size
        position_ratio = abs(self.position) / self.max_position
        risk_penalty = -position_ratio * TradingConstants.POSITION_RATIO_PENALTY
        
        # Overtrading penalty
        if self.trade_count > TradingConstants.MAX_TRADES_PER_EPISODE:
            overtrading_penalty = TradingConstants.OVERTRADING_PENALTY
        else:
            overtrading_penalty = 0
        
        # Action-specific rewards
        if action == TradingConstants.ACTION_HOLD:
            action_reward = TradingConstants.HOLD_REWARD
        else:
            action_reward = 0
        
        total_reward = base_reward + risk_penalty + overtrading_penalty + action_reward
        
        # Update history
        self.pnl_history.append(self.portfolio_value)
        
        return total_reward


# ==================== FEATURE ENGINEERING ====================

class MarketFeatureExtractor:
    """Extract comprehensive market features for transformer input"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
        
    def extract_features(self, data):
        """Extract features from OHLCV data"""
        df = data.copy()
        
        # Price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
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
        
        # Volume features
        df['volume_ma'] = df['volume'].rolling(TradingConstants.BB_PERIOD).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Rolling statistics
        for window in TradingConstants.ROLLING_WINDOWS:
            df[f'close_ma_{window}'] = df['close'].rolling(window).mean()
            df[f'close_std_{window}'] = df['close'].rolling(window).std()
            df[f'volume_ma_{window}'] = df['volume'].rolling(window).mean()
            df[f'volume_std_{window}'] = df['volume'].rolling(window).std()
            df[f'returns_ma_{window}'] = df['returns'].rolling(window).mean()
            df[f'returns_std_{window}'] = df['returns'].rolling(window).std()
            df[f'high_ma_{window}'] = df['high'].rolling(window).mean()
            df[f'low_ma_{window}'] = df['low'].rolling(window).mean()
        
        # Momentum features
        for period in TradingConstants.MOMENTUM_PERIODS:
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
        
        # Cumulative returns
        for period in TradingConstants.CUMULATIVE_RETURN_PERIODS:
            df[f'cum_return_{period}'] = df['returns'].rolling(period).sum()
        
        # Drawdown
        df['rolling_max'] = df['close'].rolling(TradingConstants.BB_PERIOD).max()
        df['drawdown'] = (df['close'] - df['rolling_max']) / df['rolling_max']
        
        # Price change features
        for lag in TradingConstants.LAG_PERIODS:
            df[f'close_diff_{lag}'] = df['close'].diff(lag)
            
        df['high_diff_1'] = df['high'].diff(1)
        df['low_diff_1'] = df['low'].diff(1)
        
        # Volatility features
        for period in TradingConstants.VOLATILITY_PERIODS:
            df[f'realized_vol_{period}'] = df['returns'].rolling(period).std() * np.sqrt(TradingConstants.VOLATILITY_ANNUALIZATION)
            
        df['vol_change_5'] = df['realized_vol_5'].diff(5)
        df['vol_zscore_20'] = (df['realized_vol_20'] - df['realized_vol_20'].rolling(TradingConstants.Z_SCORE_PERIODS[1]).mean()) / df['realized_vol_20'].rolling(TradingConstants.Z_SCORE_PERIODS[1]).std()
        
        # Price percentiles
        for period in TradingConstants.PRICE_PERCENTILE_PERIODS:
            df[f'close_pct_{period}'] = df['close'].rolling(period).apply(
                lambda x: (x[-1] - np.min(x)) / (np.max(x) - np.min(x)) if np.max(x) != np.min(x) else 0
            )
            df[f'volume_pct_{period}'] = df['volume'].rolling(period).apply(
                lambda x: (x[-1] - np.min(x)) / (np.max(x) - np.min(x)) if np.max(x) != np.min(x) else 0
            )
        
        # Candle features
        df['candle_body'] = abs(df['close'] - df['open'])
        df['candle_range'] = df['high'] - df['low']
        df['upper_shadow'] = df['high'] - np.maximum(df['close'], df['open'])
        df['lower_shadow'] = np.minimum(df['close'], df['open']) - df['low']
        df['body_to_range'] = df['candle_body'] / (df['candle_range'] + TradingConstants.EPSILON_SMALL)
        
        # Gaps
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_pct'] = df['gap'] / df['close'].shift(1)
        
        # Technical indicators
        df['sma_5'] = ta.trend.sma_indicator(df['close'], 5)
        df['sma_10'] = ta.trend.sma_indicator(df['close'], 10)
        df['sma_20'] = ta.trend.sma_indicator(df['close'], TradingConstants.BB_PERIOD)
        df['ema_12'] = ta.trend.ema_indicator(df['close'], TradingConstants.MACD_FAST_PERIOD)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], TradingConstants.MACD_SLOW_PERIOD)
        
        # MACD
        df['macd'] = ta.trend.macd(df['close'])
        df['macd_signal'] = ta.trend.macd_signal(df['close'])
        df['macd_histogram'] = ta.trend.macd_diff(df['close'])
        
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=TradingConstants.RSI_PERIOD)
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=TradingConstants.BB_PERIOD, window_dev=TradingConstants.BB_STD_DEV)
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
        
        # Market microstructure
        df['spread'] = df['high'] - df['low']
        df['midpoint'] = (df['high'] + df['low']) / 2
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Additional features
        df['turnover_20'] = df['volume'] / (df['volume_ma_20'] + TradingConstants.EPSILON_SMALL)
        df['atr_norm'] = df['atr'] / (df['close_ma_20'] + TradingConstants.EPSILON_SMALL)
        df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int)
        df['rsi_overbought'] = (df['rsi'] > TradingConstants.RSI_OVERBOUGHT).astype(int)
        df['rsi_oversold'] = (df['rsi'] < TradingConstants.RSI_OVERSOLD).astype(int)
        df['bb_pos'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + TradingConstants.EPSILON_SMALL)
        
        # Volume spike detection
        df['volume_spike'] = (df['volume'] > TradingConstants.VOLUME_SPIKE_MULTIPLIER * df['volume_ma_20']).astype(int)
        
        # Z-score of close
        df['close_zscore_20'] = (df['close'] - df['close_ma_20']) / (df['close_std_20'] + TradingConstants.EPSILON_SMALL)
        
        # Sharpe ratio
        df['sharpe_20'] = df['returns_ma_20'] / (df['returns_std_20'] + TradingConstants.EPSILON_SMALL)
        
        # Rolling correlations
        for period in TradingConstants.CORRELATION_PERIODS:
            df[f'corr_close_volume_{period}'] = df['close'].rolling(period).corr(df['volume'])
        
        # Rolling skew/kurtosis
        for period in TradingConstants.SKEW_KURTOSIS_PERIODS:
            df[f'skew_{period}'] = df['returns'].rolling(period).skew()
            df[f'kurt_{period}'] = df['returns'].rolling(period).kurt()
        
        # Time-based features
        df['hour'] = df.index.hour if hasattr(df.index, 'hour') else 0
        df['day_of_week'] = df.index.dayofweek if hasattr(df.index, 'dayofweek') else 0
        df['month'] = df.index.month if hasattr(df.index, 'month') else 0
        df['quarter'] = df.index.quarter if hasattr(df.index, 'quarter') else 0
        
        # Intraday features
        if hasattr(df.index, 'hour'):
            df['is_open'] = ((df.index.hour >= TradingConstants.MARKET_OPEN_HOUR) & 
                           (df.index.hour <= TradingConstants.MARKET_CLOSE_HOUR)).astype(int)
            df['minute'] = df.index.minute if hasattr(df.index, 'minute') else 0
            df['time_of_day'] = df.index.hour * TradingConstants.MINUTES_PER_HOUR + df['minute']
            df['time_pct'] = df['time_of_day'] / (TradingConstants.MARKET_CLOSE_HOUR * TradingConstants.MINUTES_PER_HOUR)
        
        # Lag features
        for lag in TradingConstants.LAG_PERIODS:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # External features (placeholders)
        df['news_sentiment'] = TradingConstants.SENTIMENT_FILL_VALUE
        df['trump_tweet_sentiment'] = TradingConstants.SENTIMENT_FILL_VALUE
        
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


# ==================== PORTFOLIO MANAGEMENT ====================

class HahnPortfolioManager:
    """Manage Chris Hahn-style portfolio with algorithmic execution"""
    
    def __init__(self, initial_capital=HahnPortfolioConstants.DEFAULT_INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.hahn_universe = HahnPortfolioConstants.PORTFOLIO_UNIVERSE
        self.capital_phases = HahnPortfolioConstants.CAPITAL_PHASES
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
                env = TradingEnvironment(
                    features, 
                    initial_capital=self.initial_capital * self.hahn_universe[symbol]['target_weight']
                )
                
                # Create DQN agent
                agent = DQNAgent(
                    state_dim=features.shape[1],
                    action_dim=TradingConstants.ACTION_SPACE_SIZE,
                    lr=TradingConstants.DQN_LEARNING_RATE,
                    gamma=TradingConstants.DQN_GAMMA,
                    epsilon=TradingConstants.DQN_EPSILON
                )
                
                self.scalping_agents[symbol] = {
                    'agent': agent,
                    'env': env,
                    'extractor': extractor,
                    'data': data
                }
    
    def fetch_market_data(self, symbol, period=TradingConstants.DEFAULT_PERIOD):
        """Fetch market data for symbol"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=TradingConstants.DEFAULT_INTERVAL)
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def determine_phase(self):
        """Determine current capital phase"""
        for phase, config in self.capital_phases.items():
            if config['range'][0] <= self.initial_capital <= config['range'][1]:
                return phase
        return 'phase_4'  # Default to highest phase


# ==================== DQN AGENT ====================

class DQNAgent:
    """Deep Q-Network agent with experience replay and target network"""
    
    def __init__(self, state_dim, 
                 action_dim=TradingConstants.ACTION_SPACE_SIZE,
                 lr=TradingConstants.DQN_LEARNING_RATE,
                 gamma=TradingConstants.DQN_GAMMA,
                 epsilon=TradingConstants.DQN_EPSILON):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Networks
        self.q_network = DuelingDQN(state_dim, action_dim=action_dim)
        self.target_network = DuelingDQN(state_dim, action_dim=action_dim)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(TradingConstants.DQN_REPLAY_BUFFER_SIZE)
        
        # Training parameters
        self.batch_size = TradingConstants.DQN_BATCH_SIZE
        self.update_frequency = TradingConstants.DQN_UPDATE_FREQUENCY
        self.target_update_frequency = TradingConstants.DQN_TARGET_UPDATE_FREQUENCY
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
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), TradingConstants.DQN_GRADIENT_CLIP)
        self.optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(TradingConstants.DQN_MIN_EPSILON, 
                          self.epsilon * TradingConstants.DQN_EPSILON_DECAY)
        
        return loss.item()


# ==================== REPLAY BUFFER ====================

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
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


# ==================== PERFORMANCE TRACKING ====================

class PerformanceTracker:
    """Track and analyze trading performance"""
    
    def __init__(self):
        self.symbol_performance = {}
        self.session_history = []
        
    def initialize(self):
        """Initialize performance tracking"""
        self.symbol_performance = {}
        self.session_history = []
    
    def update_symbol_performance(self, symbol, results):
        """Update performance for a specific symbol"""
        if symbol not in self.symbol_performance:
            self.symbol_performance[symbol] = []
        
        self.symbol_performance[symbol].append(results)
    
    def get_symbol_stats(self, symbol):
        """Get statistics for a symbol"""
        if symbol not in self.symbol_performance:
            return None
        
        performances = self.symbol_performance[symbol]
        total_returns = [p['trading']['total_reward'] for p in performances]
        
        return {
            'mean_return': np.mean(total_returns),
            'std_return': np.std(total_returns),
            'sharpe_ratio': np.mean(total_returns) / (np.std(total_returns) + TradingConstants.EPSILON_SMALL),
            'max_return': np.max(total_returns),
            'min_return': np.min(total_returns)
        }


# ==================== MAIN TRIDENT SYSTEM ====================

class TridentTradingSystem:
    """
    Complete Trident trading system integrating:
    - Chris Hahn portfolio selection
    - Transformer-based market analysis
    - Deep Q-learning for execution
    - Risk management and capital scaling
    """
    
    def __init__(self, initial_capital=HahnPortfolioConstants.DEFAULT_INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.portfolio_manager = HahnPortfolioManager(initial_capital)
        self.performance_tracker = PerformanceTracker()
        self.current_phase = self.portfolio_manager.determine_phase()
        
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
        training_results = self.train_agent(agent, env, episodes=TradingConstants.TRAINING_EPISODES)
        
        # Live trading simulation
        print(f"💹 Live trading {symbol}...")
        trading_results = self.simulate_live_trading(agent, env, duration_hours)
        
        return {
            'training': training_results,
            'trading': trading_results,
            'symbol': symbol
        }
    
    def train_agent(self, agent, env, episodes=TradingConstants.TRAINING_EPISODES):
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
            
            if episode % TradingConstants.TRAINING_PROGRESS_INTERVAL == 0:
                avg_reward = np.mean(episode_rewards[-TradingConstants.TRAINING_PROGRESS_INTERVAL:])
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
            print(f"  Final: ${final_value:,.2f}")
            print(f"  PnL: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
            print(f"  Trades: {trading_results['total_trades']}")
        
        total_pnl_pct = (total_pnl / self.initial_capital) * 100
        print(f"\n🎯 TOTAL RESULTS:")
        print(f"  Total PnL: ${total_pnl:,.2f} ({total_pnl_pct:+.2f}%)")
        print(f"  Total Trades: {total_trades}")
        print(f"  Average per Trade: ${total_pnl/max(total_trades, 1):,.2f}")
        
        return {
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'total_trades': total_trades,
            'symbol_results': session_results
        }


# ==================== MAIN ENTRY POINT ====================

if __name__ == "__main__":
    # Initialize system
    system = TridentTradingSystem(initial_capital=500000)
    system.initialize_system()
    
    # Run trading session
    results = system.run_trading_session(duration_hours=6)
    
    # Print final results
    print("\n✅ Trading session completed!")
    print(f"🎯 Final results: {results}")
