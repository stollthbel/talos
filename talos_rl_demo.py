#!/usr/bin/env python3
# talos_rl_demo.py — Complete RL Agent Demo with Real Tick Data
import os
import sys
import time
import numpy as np
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from talos_signal_engine import TalosSignalEngine, TalosRLAgent, Tick, Genome
from data_sources import DataManager, YahooFinanceSource, CSVDataSource

class TalosLiveTrader:
    """Live trading system with RL agent"""
    
    def __init__(self, symbol='SPY', model_path=None):
        self.symbol = symbol
        self.signal_engine = TalosSignalEngine()
        self.rl_agent = TalosRLAgent(algorithm='PPO', model_path=model_path)
        self.data_manager = DataManager()
        self.position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = 0
        self.balance = 10000
        self.trades = []
        
        # Setup data source
        yahoo_source = YahooFinanceSource(symbol)
        self.data_manager.add_source('yahoo', yahoo_source)
        
    def on_tick_received(self, tick: Tick):
        """Handle incoming tick data"""
        print(f"📊 Tick: {tick.symbol} ${tick.last:.2f} Vol: {tick.volume}")
        
        # Process tick through signal engine
        signal = self.signal_engine.process_tick(tick)
        
        if signal:
            print(f"🔥 Signal: {signal[0]} at ${signal[2]:.2f}, TP: ${signal[3]:.2f}, SL: ${signal[4]:.2f}")
        
        # Get RL agent prediction
        if self.rl_agent.model is not None:
            observation = self.signal_engine.get_state_vector()
            # Add position info
            position_info = np.array([self.balance / 10000, self.position])
            full_observation = np.concatenate([observation, position_info])
            
            action = self.rl_agent.predict(full_observation)
            self.execute_action(action, tick)
    
    def execute_action(self, action: int, tick: Tick):
        """Execute trading action"""
        current_price = tick.last
        
        if action == 1 and self.position == 0:  # Open Long
            self.position = 1
            self.entry_price = current_price
            print(f"🟢 OPENED LONG at ${current_price:.2f}")
            
        elif action == 2 and self.position == 0:  # Open Short
            self.position = -1
            self.entry_price = current_price
            print(f"🔴 OPENED SHORT at ${current_price:.2f}")
            
        elif action == 3 and self.position != 0:  # Close Position
            if self.position == 1:  # Close Long
                pnl = current_price - self.entry_price
                print(f"🟢 CLOSED LONG at ${current_price:.2f}, PnL: ${pnl:.2f}")
            else:  # Close Short
                pnl = self.entry_price - current_price
                print(f"🔴 CLOSED SHORT at ${current_price:.2f}, PnL: ${pnl:.2f}")
            
            self.balance += pnl
            self.trades.append({
                'entry_price': self.entry_price,
                'exit_price': current_price,
                'pnl': pnl,
                'position': 'LONG' if self.position == 1 else 'SHORT',
                'timestamp': tick.timestamp
            })
            
            self.position = 0
            self.entry_price = 0
            
            print(f"💰 Balance: ${self.balance:.2f}")
    
    def start_live_trading(self):
        """Start live trading"""
        print(f"🚀 Starting live trading for {self.symbol}")
        self.data_manager.start_feed('yahoo', self.on_tick_received)
    
    def stop_trading(self):
        """Stop trading"""
        self.data_manager.stop_all_feeds()
        print(f"🛑 Trading stopped. Final balance: ${self.balance:.2f}")
        print(f"📈 Total trades: {len(self.trades)}")

def train_rl_agent_demo():
    """Demo: Train RL agent on historical data"""
    print("🧠 Training RL Agent Demo")
    print("=" * 40)
    
    # Get historical data
    print("📥 Fetching historical data...")
    data_manager = DataManager()
    ticks = data_manager.create_sample_data('SPY', days=30)
    
    if len(ticks) < 1000:
        print("❌ Not enough data for training. Need at least 1000 ticks.")
        return None
    
    print(f"✅ Loaded {len(ticks)} ticks")
    
    # Split data for training and testing
    split_point = int(len(ticks) * 0.8)
    train_ticks = ticks[:split_point]
    test_ticks = ticks[split_point:]
    
    print(f"📊 Training on {len(train_ticks)} ticks, testing on {len(test_ticks)} ticks")
    
    # Train agent
    agent = TalosRLAgent(algorithm='PPO', model_path='talos_rl_model_demo')
    agent.train(train_ticks, timesteps=10000)
    
    # Backtest
    print("🔬 Running backtest...")
    total_reward, final_balance = agent.backtest(test_ticks)
    
    print(f"📈 Backtest Results:")
    print(f"   Total Reward: {total_reward:.2f}")
    print(f"   Final Balance: ${final_balance:.2f}")
    print(f"   Return: {((final_balance - 10000) / 10000) * 100:.2f}%")
    
    return agent

def signal_engine_demo():
    """Demo: Test signal engine on historical data"""
    print("🔥 Signal Engine Demo")
    print("=" * 30)
    
    # Get sample data
    data_manager = DataManager()
    ticks = data_manager.create_sample_data('SPY', days=5)
    
    if not ticks:
        print("❌ No data available")
        return
    
    print(f"📊 Processing {len(ticks)} ticks...")
    
    # Test signal engine
    engine = TalosSignalEngine()
    signals = []
    
    for tick in ticks:
        signal = engine.process_tick(tick)
        if signal:
            signals.append(signal)
            print(f"🔥 {signal[0]} at ${signal[2]:.2f} (TP: ${signal[3]:.2f}, SL: ${signal[4]:.2f})")
    
    print(f"✅ Generated {len(signals)} signals from {len(ticks)} ticks")
    print(f"📈 Signal rate: {(len(signals) / len(ticks)) * 100:.2f}%")

def data_sources_demo():
    """Demo: Test different data sources"""
    print("📡 Data Sources Demo")
    print("=" * 25)
    
    # Yahoo Finance
    print("🟡 Testing Yahoo Finance...")
    yahoo_source = YahooFinanceSource('SPY')
    yahoo_data = yahoo_source.get_historical_data(period='1d')
    print(f"   ✅ Yahoo: {len(yahoo_data)} ticks")
    
    if yahoo_data:
        latest = yahoo_data[-1]
        print(f"   📊 Latest: ${latest.last:.2f} at {datetime.fromtimestamp(latest.timestamp)}")
    
    # CSV source (if file exists)
    print("📄 Testing CSV source...")
    if os.path.exists('sample_data.csv'):
        csv_source = CSVDataSource('sample_data.csv')
        csv_data = csv_source.get_historical_data()
        print(f"   ✅ CSV: {len(csv_data)} ticks")
    else:
        print("   ⚠️  CSV file not found (sample_data.csv)")

def create_sample_csv():
    """Create sample CSV data for testing"""
    print("📝 Creating sample CSV data...")
    
    data_manager = DataManager()
    ticks = data_manager.create_sample_data('SPY', days=1)
    
    if ticks:
        import pandas as pd
        
        # Convert to DataFrame
        data = []
        for tick in ticks:
            data.append({
                'timestamp': datetime.fromtimestamp(tick.timestamp),
                'symbol': tick.symbol,
                'close': tick.last,
                'high': tick.ask,
                'low': tick.bid,
                'volume': tick.volume
            })
        
        df = pd.DataFrame(data)
        df.to_csv('sample_data.csv', index=False)
        print(f"✅ Created sample_data.csv with {len(data)} rows")
    else:
        print("❌ Failed to create sample data")

def main():
    """Main demo function"""
    print("🏛️ TALOS CAPITAL RL AGENT DEMO")
    print("=" * 50)
    
    while True:
        print("\nChoose a demo:")
        print("1. 🔥 Signal Engine Demo")
        print("2. 🧠 Train RL Agent Demo")
        print("3. 📡 Data Sources Demo")
        print("4. 📝 Create Sample CSV")
        print("5. 🚀 Live Trading Simulation")
        print("0. ❌ Exit")
        
        choice = input("\nEnter your choice (0-5): ")
        
        if choice == '1':
            signal_engine_demo()
        elif choice == '2':
            train_rl_agent_demo()
        elif choice == '3':
            data_sources_demo()
        elif choice == '4':
            create_sample_csv()
        elif choice == '5':
            print("🚀 Starting live trading simulation...")
            print("   (This will run for 2 minutes with Yahoo Finance data)")
            
            trader = TalosLiveTrader('SPY')
            
            # Try to load pre-trained model
            if os.path.exists('talos_rl_model_demo.zip'):
                trader.rl_agent.load_model('talos_rl_model_demo')
                print("✅ Loaded pre-trained RL model")
            else:
                print("⚠️  No pre-trained model found. Train one first (option 2)")
            
            trader.start_live_trading()
            
            try:
                time.sleep(120)  # Run for 2 minutes
            except KeyboardInterrupt:
                print("\n🛑 Interrupted by user")
            
            trader.stop_trading()
            
        elif choice == '0':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
