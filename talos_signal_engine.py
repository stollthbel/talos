# talos_signal_engine.py — Advanced Trading Signal Engine with RL Agent
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import json
import sqlite3
from datetime import datetime
import yfinance as yf
import alpaca_trade_api as tradeapi
import websocket
import threading
import time
import gym
from gym import spaces
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
import pickle
import os

@dataclass
class Tick:
    bid: float
    ask: float
    last: float
    volume: float
    timestamp: float
    symbol: str = "SPY"

@dataclass
class State:
    ema_fast: float
    ema_slow: float
    vwap: float
    price: float
    momentum: float
    rsi: float
    atr: float
    dprice_dt: float
    dp_dt: float
    gain: float
    loss: float
    tr: float
    rs: float
    vol_ema: float
    price_volatility: float
    take_profit: float
    stop_loss: float
    fitness: float
    fib_support: float
    fib_resistance: float
    acceleration: float
    jerk: float

@dataclass
class Genome:
    alpha_fast: float = 2. / 10.
    alpha_slow: float = 2. / 50.
    rsi_threshold: float = 30.
    take_profit_mult: float = 2.0
    stop_loss_mult: float = 1.0
    fib_ratio1: float = 0.382
    fib_ratio2: float = 0.618
    reward_risk_weight: float = 1.5
    ode_accel_coeff: float = 0.1
    ode_jerk_coeff: float = 0.01

class TalosSignalEngine:
    def __init__(self, genome: Genome = None):
        self.genome = genome or Genome()
        self.state = None
        self.tick_buffer = []
        self.signals = []
        self.cum_pv = 0.0
        self.cum_vol = 0.0
        self.high = 0.0
        self.low = float('inf')
        self.t = 0.0
        
    def ema_ode(self, alpha, ema, price):
        return alpha * (price - ema)

    def rsi_ode(self, gain, loss):
        rs = 1000.0 if loss == 0.0 else gain / loss
        return 100.0 - (100.0 / (1.0 + rs)), rs

    def vwap_ode(self, cum_pv, cum_vol, price, volume):
        tpv = price * volume
        new_cum_pv = cum_pv + tpv
        new_cum_vol = cum_vol + volume
        return new_cum_pv / new_cum_vol if new_cum_vol > 0 else price, new_cum_pv, new_cum_vol

    def true_range(self, prev_close, price, high, low):
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        return max(tr1, tr2, tr3)

    def atr_ode(self, prev_atr, tr, alpha):
        return alpha * tr + (1.0 - alpha) * prev_atr

    def volatility_ode(self, prev_vol, delta, alpha):
        return alpha * (delta * delta) + (1. - alpha) * prev_vol

    def fib_levels(self, high, low, ratio1, ratio2):
        diff = high - low
        return low + ratio1 * diff, high - ratio2 * diff

    def fitness_function(self, reward, risk, reward_risk_weight):
        return reward * reward_risk_weight if risk == 0.0 else reward / risk * reward_risk_weight

    def initialize_state(self, init_price):
        self.state = State(
            ema_fast=init_price, ema_slow=init_price, vwap=init_price, price=init_price, momentum=0.0,
            rsi=50.0, atr=0.0, dprice_dt=0.0, dp_dt=0.0, gain=0.0, loss=0.0, tr=0.0, rs=1.0,
            vol_ema=0.0, price_volatility=0.0, take_profit=init_price * 1.02, stop_loss=init_price * 0.98,
            fitness=0.0, fib_support=init_price * 0.9, fib_resistance=init_price * 1.1, acceleration=0.0, jerk=0.0
        )
        self.high = init_price
        self.low = init_price

    def process_tick(self, tick: Tick):
        if self.state is None:
            self.initialize_state(tick.last)
            return None
            
        prev_close = self.state.price
        self.high = max(self.high, tick.last)
        self.low = min(self.low, tick.last)
        
        # Update state using ODE
        self.state, (self.cum_pv, self.cum_vol) = self.state_ode(
            self.genome, self.t, self.state, tick, prev_close, 
            self.cum_pv, self.cum_vol, self.high, self.low
        )
        
        # Generate signals
        signal = self.generate_signal()
        self.t += 1.0
        
        return signal

    def state_ode(self, genome, t, st: State, tick: Tick, prev_close, cum_pv, cum_vol, high, low):
        d_ema_fast = self.ema_ode(genome.alpha_fast, st.ema_fast, tick.last)
        d_ema_slow = self.ema_ode(genome.alpha_slow, st.ema_slow, tick.last)
        new_vwap, new_pv, new_vol = self.vwap_ode(cum_pv, cum_vol, tick.last, tick.volume)
        price = tick.last
        delta = price - prev_close
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        new_gain = 0.1 * gain + 0.9 * st.gain
        new_loss = 0.1 * loss + 0.9 * st.loss
        new_rsi, rs = self.rsi_ode(new_gain, new_loss)
        tr = self.true_range(prev_close, price, high, low)
        new_tr = 0.1 * tr + 0.9 * st.tr
        new_atr = self.atr_ode(st.atr, new_tr, 2. / 15.)
        new_vol_ema = self.ema_ode(2. / 21., st.vol_ema, tick.volume)
        new_volatility = self.volatility_ode(st.price_volatility, delta, 2. / 21.)
        acceleration = genome.ode_accel_coeff * (delta - st.dprice_dt) + (1. - genome.ode_accel_coeff) * st.acceleration
        jerk = genome.ode_jerk_coeff * (acceleration - st.acceleration) + (1. - genome.ode_jerk_coeff) * st.jerk
        dp_dt = st.dprice_dt + acceleration + 0.1 * (st.ema_fast - price) + 0.05 * (st.vwap - price) + 0.01 * (new_volatility - st.price_volatility)
        take_profit = price + genome.take_profit_mult * new_atr
        stop_loss = price - genome.stop_loss_mult * new_atr
        fib_supp, fib_res = self.fib_levels(high, low, genome.fib_ratio1, genome.fib_ratio2)
        reward = take_profit - price
        risk = price - stop_loss
        fitness = self.fitness_function(reward, risk, genome.reward_risk_weight)
        
        return State(
            ema_fast=st.ema_fast + d_ema_fast,
            ema_slow=st.ema_slow + d_ema_slow,
            vwap=new_vwap,
            price=price + st.dprice_dt,
            momentum=dp_dt,
            rsi=new_rsi,
            atr=new_atr,
            dprice_dt=dp_dt,
            dp_dt=0.0,
            gain=new_gain,
            loss=new_loss,
            tr=new_tr,
            rs=rs,
            vol_ema=new_vol_ema,
            price_volatility=new_volatility,
            take_profit=take_profit,
            stop_loss=stop_loss,
            fitness=fitness,
            fib_support=fib_supp,
            fib_resistance=fib_res,
            acceleration=acceleration,
            jerk=jerk
        ), (new_pv, new_vol)

    def generate_signal(self):
        if self.state is None:
            return None
            
        long_signal = (
            self.state.ema_fast > self.state.ema_slow and
            self.state.price > self.state.vwap and
            self.state.rsi < self.genome.rsi_threshold and
            self.state.momentum > 0.0 and
            self.state.vol_ema > 0.0 and
            self.state.price > self.state.fib_support
        )
        short_signal = (
            self.state.ema_fast < self.state.ema_slow and
            self.state.price < self.state.vwap and
            self.state.rsi > (100. - self.genome.rsi_threshold) and
            self.state.momentum < 0.0 and
            self.state.vol_ema < 0.0 and
            self.state.price < self.state.fib_resistance
        )
        
        if long_signal:
            return ("LONG", self.t, self.state.price, self.state.take_profit, self.state.stop_loss, self.state.fitness)
        elif short_signal:
            return ("SHORT", self.t, self.state.price, self.state.take_profit, self.state.stop_loss, self.state.fitness)
        else:
            return None

    def get_state_vector(self):
        """Convert current state to vector for RL agent"""
        if self.state is None:
            return np.zeros(15)
        return np.array([
            self.state.ema_fast / self.state.price,
            self.state.ema_slow / self.state.price,
            self.state.vwap / self.state.price,
            self.state.momentum,
            self.state.rsi / 100.0,
            self.state.atr / self.state.price,
            self.state.price_volatility,
            self.state.fib_support / self.state.price,
            self.state.fib_resistance / self.state.price,
            self.state.acceleration,
            self.state.jerk,
            self.state.vol_ema / 1000.0,  # Normalize volume
            self.state.fitness,
            (self.state.take_profit - self.state.price) / self.state.price,
            (self.state.price - self.state.stop_loss) / self.state.price
        ])

class TradingEnvironment(gym.Env):
    """Custom Trading Environment for RL Agent"""
    
    def __init__(self, tick_data: List[Tick], initial_balance=10000):
        super(TradingEnvironment, self).__init__()
        
        self.tick_data = tick_data
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = 0
        self.signal_engine = TalosSignalEngine()
        
        # Define action and observation space
        # Actions: 0 = Hold, 1 = Long, 2 = Short, 3 = Close
        self.action_space = spaces.Discrete(4)
        
        # Observation space: 15 features from signal engine + balance + position
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.signal_engine = TalosSignalEngine()
        
        if self.tick_data:
            self.signal_engine.process_tick(self.tick_data[0])
        
        return self._get_observation()
    
    def _get_observation(self):
        state_vector = self.signal_engine.get_state_vector()
        position_info = np.array([self.balance / self.initial_balance, self.position])
        return np.concatenate([state_vector, position_info])
    
    def step(self, action):
        if self.current_step >= len(self.tick_data) - 1:
            return self._get_observation(), 0, True, {}
        
        self.current_step += 1
        tick = self.tick_data[self.current_step]
        
        # Process tick through signal engine
        signal = self.signal_engine.process_tick(tick)
        
        # Calculate reward
        reward = 0
        current_price = tick.last
        
        # Execute action
        if action == 1 and self.position == 0:  # Open Long
            self.position = 1
            self.entry_price = current_price
        elif action == 2 and self.position == 0:  # Open Short
            self.position = -1
            self.entry_price = current_price
        elif action == 3 and self.position != 0:  # Close Position
            if self.position == 1:  # Close Long
                pnl = current_price - self.entry_price
            else:  # Close Short
                pnl = self.entry_price - current_price
            
            self.balance += pnl
            reward = pnl / self.initial_balance * 100  # Percentage return
            self.position = 0
            self.entry_price = 0
        
        # Unrealized PnL reward for open positions
        if self.position != 0:
            if self.position == 1:
                unrealized_pnl = current_price - self.entry_price
            else:
                unrealized_pnl = self.entry_price - current_price
            reward = unrealized_pnl / self.initial_balance * 10  # Smaller reward for unrealized
        
        # Add fitness score from signal engine
        if signal:
            reward += signal[5] * 0.1  # Add 10% of fitness score
        
        # Episode done conditions
        done = (self.current_step >= len(self.tick_data) - 1 or 
                self.balance <= self.initial_balance * 0.1)  # Stop loss at 90% loss
        
        return self._get_observation(), reward, done, {}
    
    def render(self, mode='human'):
        profit = self.balance - self.initial_balance
        print(f"Step: {self.current_step}, Balance: ${self.balance:.2f}, "
              f"Profit: ${profit:.2f}, Position: {self.position}")

class TalosRLAgent:
    """Reinforcement Learning Agent for Trading"""
    
    def __init__(self, algorithm='PPO', model_path=None):
        self.algorithm = algorithm
        self.model_path = model_path
        self.model = None
        self.env = None
        
    def create_environment(self, tick_data: List[Tick]):
        """Create training environment"""
        self.env = TradingEnvironment(tick_data)
        return self.env
    
    def train(self, tick_data: List[Tick], timesteps=10000):
        """Train the RL agent"""
        env = self.create_environment(tick_data)
        
        if self.algorithm == 'PPO':
            self.model = PPO('MlpPolicy', env, verbose=1)
        elif self.algorithm == 'A2C':
            self.model = A2C('MlpPolicy', env, verbose=1)
        elif self.algorithm == 'DQN':
            self.model = DQN('MlpPolicy', env, verbose=1)
        
        print(f"Training {self.algorithm} agent for {timesteps} timesteps...")
        self.model.learn(total_timesteps=timesteps)
        
        if self.model_path:
            self.model.save(self.model_path)
            print(f"Model saved to {self.model_path}")
    
    def load_model(self, model_path):
        """Load pre-trained model"""
        if self.algorithm == 'PPO':
            self.model = PPO.load(model_path)
        elif self.algorithm == 'A2C':
            self.model = A2C.load(model_path)
        elif self.algorithm == 'DQN':
            self.model = DQN.load(model_path)
        
        print(f"Model loaded from {model_path}")
    
    def predict(self, observation):
        """Predict action given observation"""
        if self.model is None:
            return 0  # Default to hold
        
        action, _ = self.model.predict(observation)
        return action
    
    def backtest(self, tick_data: List[Tick]):
        """Backtest the trained agent"""
        env = self.create_environment(tick_data)
        obs = env.reset()
        total_reward = 0
        
        for _ in range(len(tick_data) - 1):
            action = self.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            
            if done:
                break
        
        return total_reward, env.balance

class RealTimeDataFeeder:
    """Real-time data feeder for live trading"""
    
    def __init__(self, symbol='SPY', data_source='yahoo'):
        self.symbol = symbol
        self.data_source = data_source
        self.callbacks = []
        self.running = False
        
    def add_callback(self, callback):
        """Add callback function to receive tick data"""
        self.callbacks.append(callback)
    
    def start_yahoo_feed(self):
        """Start Yahoo Finance data feed (1-minute intervals)"""
        self.running = True
        
        def feed_loop():
            while self.running:
                try:
                    # Get latest 1-minute data
                    ticker = yf.Ticker(self.symbol)
                    data = ticker.history(period="1d", interval="1m")
                    
                    if not data.empty:
                        latest = data.iloc[-1]
                        tick = Tick(
                            bid=latest['Low'],
                            ask=latest['High'],
                            last=latest['Close'],
                            volume=latest['Volume'],
                            timestamp=time.time(),
                            symbol=self.symbol
                        )
                        
                        for callback in self.callbacks:
                            callback(tick)
                    
                    time.sleep(60)  # Wait 1 minute
                except Exception as e:
                    print(f"Error fetching data: {e}")
                    time.sleep(5)
        
        thread = threading.Thread(target=feed_loop)
        thread.daemon = True
        thread.start()
    
    def start_alpaca_feed(self, api_key, secret_key, base_url):
        """Start Alpaca data feed (real-time)"""
        try:
            import alpaca_trade_api as tradeapi
            
            api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
            
            def on_minute_bar(bar):
                tick = Tick(
                    bid=bar.low,
                    ask=bar.high,
                    last=bar.close,
                    volume=bar.volume,
                    timestamp=bar.timestamp.timestamp(),
                    symbol=bar.symbol
                )
                
                for callback in self.callbacks:
                    callback(tick)
            
            # Subscribe to minute bars
            conn = tradeapi.StreamConn(api_key, secret_key, base_url)
            conn.on(r'AM\.' + self.symbol)(on_minute_bar)
            conn.run()
            
        except ImportError:
            print("Alpaca Trade API not installed. Please install: pip install alpaca-trade-api")
    
    def stop(self):
        """Stop data feed"""
        self.running = False

def load_csv_data(file_path: str) -> List[Tick]:
    """Load tick data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        ticks = []
        
        for _, row in df.iterrows():
            tick = Tick(
                bid=row.get('bid', row.get('low', row.get('close', 0))),
                ask=row.get('ask', row.get('high', row.get('close', 0))),
                last=row.get('last', row.get('close', 0)),
                volume=row.get('volume', 0),
                timestamp=pd.to_datetime(row.get('timestamp', row.get('date', '2024-01-01'))).timestamp(),
                symbol=row.get('symbol', 'SPY')
            )
            ticks.append(tick)
        
        return ticks
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        return []

def create_sample_data(symbol='SPY', days=30) -> List[Tick]:
    """Create sample tick data using Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=f"{days}d", interval="1m")
        
        ticks = []
        for timestamp, row in data.iterrows():
            tick = Tick(
                bid=row['Low'],
                ask=row['High'],
                last=row['Close'],
                volume=row['Volume'],
                timestamp=timestamp.timestamp(),
                symbol=symbol
            )
            ticks.append(tick)
        
        return ticks
    except Exception as e:
        print(f"Error creating sample data: {e}")
        return []

if __name__ == "__main__":
    # Example usage
    print("🏛️ Talos Signal Engine with RL Agent")
    print("=" * 50)
    
    # Create sample data
    print("Creating sample data...")
    ticks = create_sample_data('SPY', days=5)
    print(f"Loaded {len(ticks)} ticks")
    
    # Test signal engine
    print("\nTesting signal engine...")
    engine = TalosSignalEngine()
    signals = []
    
    for i, tick in enumerate(ticks[:100]):  # Test first 100 ticks
        signal = engine.process_tick(tick)
        if signal:
            signals.append(signal)
            print(f"Signal: {signal[0]} at {signal[1]:.0f}, Price: ${signal[2]:.2f}")
    
    print(f"\nGenerated {len(signals)} signals from {100} ticks")
    
    # Train RL agent
    if len(ticks) > 1000:
        print("\nTraining RL agent...")
        agent = TalosRLAgent(algorithm='PPO', model_path='talos_rl_model')
        agent.train(ticks[:1000], timesteps=5000)
        
        # Backtest
        print("\nBacktesting...")
        total_reward, final_balance = agent.backtest(ticks[1000:1500])
        print(f"Backtest Results: Total Reward: {total_reward:.2f}, Final Balance: ${final_balance:.2f}")
    
    print("\n✅ Setup complete! Ready for live trading.")
