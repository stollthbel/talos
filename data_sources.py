# data_sources.py — Real-time and Historical Data Sources for Talos Capital
from typing import List, Dict, Optional, Any
from datetime import datetime
import yfinance as yf
import requests
import json
import pandas as pd
import numpy as np
import websocket
import threading
import time
from datetime import datetime, timedelta
from typing import List, Dict, Callable, Optional
import requests
import sqlite3
from talos_signal_engine import Tick

class DataSource:
    """Base class for data sources"""
    
    def __init__(self, symbol: str = 'SPY'):
        self.symbol = symbol
        self.callbacks = []
        self.running = False
        
    def add_callback(self, callback: Callable):
        self.callbacks.append(callback)
        
    def notify_callbacks(self, tick: Tick):
        for callback in self.callbacks:
            callback(tick)
            
    def start(self):
        raise NotImplementedError
        
    def stop(self):
        self.running = False

class YahooFinanceSource(DataSource):
    """Yahoo Finance data source - Free, 1-minute intervals"""
    
    def __init__(self, symbol: str = 'SPY', interval: str = '1m'):
        super().__init__(symbol)
        self.interval = interval
        self.last_timestamp = None
        
    def get_historical_data(self, period: str = '1d') -> List[Tick]:
        """Get historical data"""
        try:
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period=period, interval=self.interval)
            
            ticks = []
            for timestamp, row in data.iterrows():
                tick = Tick(
                    bid=row['Low'],
                    ask=row['High'],
                    last=row['Close'],
                    volume=row['Volume'],
                    timestamp=timestamp.timestamp(),
                    symbol=self.symbol
                )
                ticks.append(tick)
            
            return ticks
        except Exception as e:
            print(f"Error fetching Yahoo data: {e}")
            return []
    
    def start(self):
        """Start real-time data feed (polling every minute)"""
        self.running = True
        
        def feed_loop():
            while self.running:
                try:
                    # Get latest data
                    data = self.get_historical_data(period='1d')
                    
                    if data:
                        latest_tick = data[-1]
                        
                        # Only send if it's a new timestamp
                        if self.last_timestamp is None or latest_tick.timestamp > self.last_timestamp:
                            self.last_timestamp = latest_tick.timestamp
                            self.notify_callbacks(latest_tick)
                    
                    time.sleep(60)  # Wait 1 minute
                except Exception as e:
                    print(f"Error in Yahoo feed: {e}")
                    time.sleep(5)
        
        thread = threading.Thread(target=feed_loop)
        thread.daemon = True
        thread.start()
        print(f"Yahoo Finance feed started for {self.symbol}")
    
    def get_current_price(self, symbol: str = "SPY") -> Dict[str, Any]:
        """Get current price for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d", interval="1m")
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                volume = hist['Volume'].iloc[-1]
                
                # Calculate change
                if len(hist) > 1:
                    prev_price = hist['Close'].iloc[-2]
                    change_percent = ((current_price - prev_price) / prev_price) * 100
                else:
                    change_percent = 0.0
                
                return {
                    'price': float(current_price),
                    'volume': float(volume),
                    'change_percent': float(change_percent),
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'high_24h': float(hist['High'].max()),
                    'low_24h': float(hist['Low'].min()),
                    'avg_volume': float(hist['Volume'].mean())
                }
            else:
                return self._get_fallback_data(symbol)
                
        except Exception as e:
            print(f"Error getting price for {symbol}: {e}")
            return self._get_fallback_data(symbol)
    
    def _get_fallback_data(self, symbol: str) -> Dict[str, Any]:
        """Return fallback data when API fails"""
        return {
            'price': 100.0,
            'volume': 1000.0,
            'change_percent': 0.0,
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'high_24h': 101.0,
            'low_24h': 99.0,
            'avg_volume': 1000.0
        }
    
    def _get_fallback_data(self, symbol: str) -> Dict[str, Any]:
        """Fallback data when API fails"""
        return {
            'price': 100.0 + hash(symbol) % 50,  # Pseudo-random price
            'volume': 1000000,
            'change_percent': (hash(symbol) % 200 - 100) / 100,  # -1% to +1%
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'high_24h': 105.0,
            'low_24h': 95.0,
            'avg_volume': 1000000
        }

class AlpacaDataSource(DataSource):
    """Alpaca Markets data source - Real-time, requires API key"""
     def __init__(self, symbol: str = 'SPY', api_key: Optional[str] = None, secret_key: Optional[str] = None,
                 base_url: str = 'https://paper-api.alpaca.markets'):
        super().__init__(symbol)
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        if api_key and secret_key:
            self.headers = {
                'APCA-API-KEY-ID': api_key,
                'APCA-API-SECRET-KEY': secret_key
            }
        else:
            self.headers = {}
        
    def get_historical_data(self, timeframe: str = '1Min', limit: int = 1000) -> List[Tick]:
        """Get historical bars from Alpaca"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=5)
            
            url = f"{self.base_url}/v2/stocks/{self.symbol}/bars"
            params = {
                'timeframe': timeframe,
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'limit': limit
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            data = response.json()
            
            ticks = []
            if 'bars' in data:
                for bar in data['bars']:
                    tick = Tick(
                        bid=bar['l'],  # Low as bid
                        ask=bar['h'],  # High as ask
                        last=bar['c'],  # Close as last
                        volume=bar['v'],
                        timestamp=pd.to_datetime(bar['t']).timestamp(),
                        symbol=self.symbol
                    )
                    ticks.append(tick)
            
            return ticks
        except Exception as e:
            print(f"Error fetching Alpaca data: {e}")
            return []
    
    def start_websocket_feed(self):
        """Start real-time websocket feed"""
        try:
            import alpaca_trade_api as tradeapi
            
            api = tradeapi.REST(self.api_key, self.secret_key, self.base_url, api_version='v2')
            
            def on_minute_bar(bar):
                tick = Tick(
                    bid=bar.low,
                    ask=bar.high,
                    last=bar.close,
                    volume=bar.volume,
                    timestamp=bar.timestamp.timestamp(),
                    symbol=bar.symbol
                )
                self.notify_callbacks(tick)
            
            conn = tradeapi.StreamConn(self.api_key, self.secret_key, self.base_url)
            conn.on(r'AM\.' + self.symbol)(on_minute_bar)
            
            def run_stream():
                conn.run()
            
            thread = threading.Thread(target=run_stream)
            thread.daemon = True
            thread.start()
            
            print(f"Alpaca WebSocket feed started for {self.symbol}")
            
        except ImportError:
            print("Alpaca Trade API not installed. Please install: pip install alpaca-trade-api")
        except Exception as e:
            print(f"Error starting Alpaca WebSocket: {e}")

class FinnhubDataSource(DataSource):
    """Finnhub data source - Free tier available"""
    
    def __init__(self, symbol: str = 'SPY', api_key: str = None):
        super().__init__(symbol)
        self.api_key = api_key
        self.ws = None
        
    def get_historical_data(self, resolution: str = '1', days: int = 30) -> List[Tick]:
        """Get historical data from Finnhub"""
        try:
            end_time = int(datetime.now().timestamp())
            start_time = int((datetime.now() - timedelta(days=days)).timestamp())
            
            url = f"https://finnhub.io/api/v1/stock/candle"
            params = {
                'symbol': self.symbol,
                'resolution': resolution,
                'from': start_time,
                'to': end_time,
                'token': self.api_key
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            ticks = []
            if data['s'] == 'ok':
                for i in range(len(data['t'])):
                    tick = Tick(
                        bid=data['l'][i],  # Low as bid
                        ask=data['h'][i],  # High as ask
                        last=data['c'][i],  # Close as last
                        volume=data['v'][i],
                        timestamp=data['t'][i],
                        symbol=self.symbol
                    )
                    ticks.append(tick)
            
            return ticks
        except Exception as e:
            print(f"Error fetching Finnhub data: {e}")
            return []
    
    def start_websocket_feed(self):
        """Start real-time WebSocket feed"""
        def on_message(ws, message):
            try:
                data = json.loads(message)
                if 'data' in data:
                    for trade in data['data']:
                        tick = Tick(
                            bid=trade['p'],  # Use price for all
                            ask=trade['p'],
                            last=trade['p'],
                            volume=trade['v'],
                            timestamp=trade['t'] / 1000,  # Convert to seconds
                            symbol=trade['s']
                        )
                        self.notify_callbacks(tick)
            except Exception as e:
                print(f"Error processing message: {e}")
        
        def on_error(ws, error):
            print(f"WebSocket error: {error}")
        
        def on_close(ws):
            print("WebSocket connection closed")
        
        def on_open(ws):
            # Subscribe to real-time trades
            ws.send(json.dumps({'type': 'subscribe', 'symbol': self.symbol}))
            print(f"Finnhub WebSocket feed started for {self.symbol}")
        
        websocket.enableTrace(True)
        ws_url = f"wss://ws.finnhub.io?token={self.api_key}"
        self.ws = websocket.WebSocketApp(ws_url,
                                        on_message=on_message,
                                        on_error=on_error,
                                        on_close=on_close,
                                        on_open=on_open)
        
        def run_websocket():
            self.ws.run_forever()
        
        thread = threading.Thread(target=run_websocket)
        thread.daemon = True
        thread.start()

class CSVDataSource(DataSource):
    """CSV file data source for backtesting"""
    
    def __init__(self, file_path: str, symbol: str = 'SPY'):
        super().__init__(symbol)
        self.file_path = file_path
        
    def get_historical_data(self) -> List[Tick]:
        """Load data from CSV file"""
        try:
            df = pd.read_csv(self.file_path)
            ticks = []
            
            for _, row in df.iterrows():
                tick = Tick(
                    bid=row.get('bid', row.get('low', row.get('close', 0))),
                    ask=row.get('ask', row.get('high', row.get('close', 0))),
                    last=row.get('last', row.get('close', 0)),
                    volume=row.get('volume', 0),
                    timestamp=pd.to_datetime(row.get('timestamp', row.get('date', '2024-01-01'))).timestamp(),
                    symbol=row.get('symbol', self.symbol)
                )
                ticks.append(tick)
            
            return ticks
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            return []
    
    def start_replay_feed(self, speed_multiplier: float = 1.0):
        """Replay historical data at specified speed"""
        ticks = self.get_historical_data()
        
        def replay_loop():
            self.running = True
            prev_timestamp = None
            
            for tick in ticks:
                if not self.running:
                    break
                
                # Calculate delay based on actual time differences
                if prev_timestamp is not None:
                    time_diff = tick.timestamp - prev_timestamp
                    time.sleep(max(0, time_diff / speed_multiplier))
                
                self.notify_callbacks(tick)
                prev_timestamp = tick.timestamp
        
        thread = threading.Thread(target=replay_loop)
        thread.daemon = True
        thread.start()
        
        print(f"CSV replay feed started for {self.file_path} at {speed_multiplier}x speed")

class DatabaseDataSource(DataSource):
    """SQLite database data source"""
    
    def __init__(self, db_path: str = 'trading_data.db', symbol: str = 'SPY'):
        super().__init__(symbol)
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize database table"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                bid REAL,
                ask REAL,
                last REAL,
                volume REAL,
                timestamp REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    
    def save_tick(self, tick: Tick):
        """Save tick to database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO ticks (symbol, bid, ask, last, volume, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (tick.symbol, tick.bid, tick.ask, tick.last, tick.volume, tick.timestamp))
        conn.commit()
        conn.close()
    
    def get_historical_data(self, limit: int = 1000) -> List[Tick]:
        """Get historical data from database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            SELECT symbol, bid, ask, last, volume, timestamp
            FROM ticks
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (self.symbol, limit))
        
        rows = c.fetchall()
        conn.close()
        
        ticks = []
        for row in rows:
            tick = Tick(
                symbol=row[0],
                bid=row[1],
                ask=row[2],
                last=row[3],
                volume=row[4],
                timestamp=row[5]
            )
            ticks.append(tick)
        
        return list(reversed(ticks))  # Return in chronological order

class DataManager:
    """Central data manager for all data sources"""
    
    def __init__(self):
        self.sources = {}
        self.active_feeds = []
        
    def add_source(self, name: str, source: DataSource):
        """Add a data source"""
        self.sources[name] = source
        
    def get_source(self, name: str) -> Optional[DataSource]:
        """Get a data source by name"""
        return self.sources.get(name)
    
    def start_feed(self, source_name: str, callback: Callable):
        """Start a data feed"""
        source = self.get_source(source_name)
        if source:
            source.add_callback(callback)
            source.start()
            self.active_feeds.append(source)
            print(f"Started feed: {source_name}")
        else:
            print(f"Source not found: {source_name}")
    
    def stop_all_feeds(self):
        """Stop all active feeds"""
        for source in self.active_feeds:
            source.stop()
        self.active_feeds.clear()
        print("All feeds stopped")
    
    def create_sample_data(self, symbol: str = 'SPY', days: int = 30) -> List[Tick]:
        """Create sample data using Yahoo Finance"""
        yahoo_source = YahooFinanceSource(symbol)
        return yahoo_source.get_historical_data(period=f'{days}d')

# Example usage and testing
if __name__ == "__main__":
    print("🔄 Talos Data Sources Test")
    print("=" * 40)
    
    # Initialize data manager
    manager = DataManager()
    
    # Add Yahoo Finance source
    yahoo_source = YahooFinanceSource('SPY')
    manager.add_source('yahoo', yahoo_source)
    
    # Add CSV source (if file exists)
    csv_source = CSVDataSource('sample_data.csv')
    manager.add_source('csv', csv_source)
    
    # Test callback function
    def on_tick_received(tick: Tick):
        print(f"Received tick: {tick.symbol} ${tick.last:.2f} Vol: {tick.volume}")
    
    # Get historical data
    print("\nFetching historical data...")
    historical_ticks = manager.create_sample_data('SPY', days=1)
    print(f"Loaded {len(historical_ticks)} historical ticks")
    
    if historical_ticks:
        print(f"First tick: {historical_ticks[0].symbol} ${historical_ticks[0].last:.2f}")
        print(f"Last tick: {historical_ticks[-1].symbol} ${historical_ticks[-1].last:.2f}")
    
    # Test real-time feed (uncomment to test)
    # print("\nStarting real-time feed...")
    # manager.start_feed('yahoo', on_tick_received)
    # time.sleep(120)  # Run for 2 minutes
    # manager.stop_all_feeds()
    
    print("\n✅ Data sources test complete!")
