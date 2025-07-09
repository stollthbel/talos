#!/usr/bin/env python3
"""
🏛️ TALOS CAPITAL MASTER ORCHESTRATOR 🏛️
The Supreme Command Center for the Stoll AI Trading Empire

Integrates:
- OCaml SignalNet evolutionary engine
- Python Stoll AI agents
- Real-time data sources
- Multi-agent coordination
- Advanced portfolio management
- Risk management systems
"""

import asyncio
import subprocess
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import websockets
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path

# Import our custom modules
from stoll_agents import PersonalizedAgent
from data_sources import AlpacaDataSource, YFinanceDataSource
from stoll_ai import StollAI

@dataclass
class TradingSignal:
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    entry_price: float
    take_profit: float
    stop_loss: float
    source: str  # 'ocaml_signalnet', 'stoll_ai', 'hybrid'
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class PortfolioPosition:
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    entry_time: datetime
    runner_mode: bool = False

class TalosMasterOrchestrator:
    """The central nervous system of the Talos trading empire"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.ocaml_process = None
        self.stoll_ai = StollAI()
        self.agents = {}
        self.data_sources = {}
        self.portfolio = {}
        self.active_signals = []
        self.performance_metrics = {}
        self.running = False
        
        # Initialize components
        self._initialize_data_sources()
        self._initialize_agents()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger('TalosMaster')
        logger.setLevel(logging.INFO)
        
        # Create handlers
        file_handler = logging.FileHandler('talos_master.log')
        console_handler = logging.StreamHandler()
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _initialize_data_sources(self):
        """Initialize all data sources"""
        self.logger.info("🔌 Initializing data sources...")
        
        # YFinance for historical data
        self.data_sources['yfinance'] = YFinanceDataSource()
        
        # Alpaca for real-time data (if API keys available)
        try:
            self.data_sources['alpaca'] = AlpacaDataSource()
            self.logger.info("✅ Alpaca data source initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Alpaca initialization failed: {e}")
    
    def _initialize_agents(self):
        """Initialize Stoll AI agents"""
        self.logger.info("🤖 Initializing AI agents...")
        
        # Theo's personalized agent
        self.agents['theo'] = PersonalizedAgent(
            name="Theo",
            agent_type="executive_trading"
        )
        
        # Risk management agent
        self.agents['risk_manager'] = PersonalizedAgent(
            name="RiskGuard",
            agent_type="risk_management"
        )
        
        # Portfolio optimization agent
        self.agents['portfolio_optimizer'] = PersonalizedAgent(
            name="PortfolioMax",
            agent_type="portfolio_optimization"
        )
        
        self.logger.info(f"✅ Initialized {len(self.agents)} AI agents")
    
    async def start_ocaml_signalnet(self):
        """Start the OCaml SignalNet evolutionary engine"""
        self.logger.info("🧬 Starting OCaml SignalNet engine...")
        
        try:
            # Compile and run the OCaml signal engine
            ocaml_path = Path("OCaml/signalnet.ml")
            if ocaml_path.exists():
                # Compile OCaml code
                compile_result = subprocess.run(
                    ["ocamlc", "-o", "signalnet", "signalnet.ml"],
                    cwd="OCaml",
                    capture_output=True,
                    text=True
                )
                
                if compile_result.returncode == 0:
                    # Start the signal engine
                    self.ocaml_process = subprocess.Popen(
                        ["./signalnet"],
                        cwd="OCaml",
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    self.logger.info("✅ OCaml SignalNet engine started")
                    return True
                else:
                    self.logger.error(f"❌ OCaml compilation failed: {compile_result.stderr}")
                    return False
            else:
                self.logger.error("❌ OCaml SignalNet file not found")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to start OCaml engine: {e}")
            return False
    
    async def get_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """Get real-time market data for symbols"""
        market_data = {}
        
        for symbol in symbols:
            try:
                # Try Alpaca first, fallback to YFinance
                if 'alpaca' in self.data_sources:
                    data = self.data_sources['alpaca'].get_current_price(symbol)
                else:
                    data = self.data_sources['yfinance'].get_current_price(symbol)
                
                market_data[symbol] = data
                
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to get data for {symbol}: {e}")
                
        return market_data
    
    async def process_ocaml_signals(self) -> List[TradingSignal]:
        """Process signals from the OCaml engine"""
        signals = []
        
        if self.ocaml_process and self.ocaml_process.poll() is None:
            try:
                # Read output from OCaml process
                line = self.ocaml_process.stdout.readline()
                if line:
                    # Parse OCaml signal output
                    if "Signal" in line and "Entry=" in line:
                        # Parse the signal format: "Signal X: Entry=Y, TP=Z, SL=W, Quality=Q"
                        parts = line.strip().split(', ')
                        
                        entry_price = float(parts[0].split('=')[1])
                        take_profit = float(parts[1].split('=')[1])
                        stop_loss = float(parts[2].split('=')[1])
                        quality = float(parts[3].split('=')[1])
                        
                        signal = TradingSignal(
                            symbol="SPY",  # Default symbol
                            signal_type="BUY" if take_profit > entry_price else "SELL",
                            confidence=quality,
                            entry_price=entry_price,
                            take_profit=take_profit,
                            stop_loss=stop_loss,
                            source="ocaml_signalnet",
                            timestamp=datetime.now(),
                            metadata={"quality": quality}
                        )
                        
                        signals.append(signal)
                        self.logger.info(f"📡 OCaml Signal: {signal.signal_type} {signal.symbol} @ {signal.entry_price}")
                        
            except Exception as e:
                self.logger.error(f"❌ Error processing OCaml signals: {e}")
        
        return signals
    
    async def generate_stoll_ai_signals(self, market_data: Dict[str, Any]) -> List[TradingSignal]:
        """Generate signals using Stoll AI agents"""
        signals = []
        
        try:
            # Get analysis from Theo's agent
            analysis = await self.agents['theo'].analyze_market_data(market_data)
            
            # Convert analysis to trading signals
            for symbol, data in market_data.items():
                if analysis.get('recommendation') in ['BUY', 'SELL']:
                    signal = TradingSignal(
                        symbol=symbol,
                        signal_type=analysis['recommendation'],
                        confidence=analysis.get('confidence', 0.5),
                        entry_price=data['price'],
                        take_profit=data['price'] * (1.02 if analysis['recommendation'] == 'BUY' else 0.98),
                        stop_loss=data['price'] * (0.99 if analysis['recommendation'] == 'BUY' else 1.01),
                        source="stoll_ai",
                        timestamp=datetime.now(),
                        metadata=analysis
                    )
                    
                    signals.append(signal)
                    self.logger.info(f"🤖 Stoll AI Signal: {signal.signal_type} {signal.symbol} @ {signal.entry_price}")
        
        except Exception as e:
            self.logger.error(f"❌ Error generating Stoll AI signals: {e}")
        
        return signals
    
    async def execute_signal(self, signal: TradingSignal) -> bool:
        """Execute a trading signal"""
        try:
            # Risk management check
            risk_check = await self.agents['risk_manager'].evaluate_signal(signal)
            
            if not risk_check.get('approved', False):
                self.logger.warning(f"🚫 Signal rejected by risk management: {risk_check.get('reason')}")
                return False
            
            # Portfolio management check
            portfolio_check = await self.agents['portfolio_optimizer'].evaluate_signal(signal)
            
            if not portfolio_check.get('approved', False):
                self.logger.warning(f"🚫 Signal rejected by portfolio optimizer: {portfolio_check.get('reason')}")
                return False
            
            # Execute the trade (simulation for now)
            position = PortfolioPosition(
                symbol=signal.symbol,
                quantity=1000.0,  # Fixed position size for now
                entry_price=signal.entry_price,
                current_price=signal.entry_price,
                unrealized_pnl=0.0,
                entry_time=signal.timestamp
            )
            
            self.portfolio[signal.symbol] = position
            self.active_signals.append(signal)
            
            self.logger.info(f"✅ Executed signal: {signal.signal_type} {signal.symbol} @ {signal.entry_price}")
            
            # Notify Theo
            await self.agents['theo'].notify_trade_execution(signal)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error executing signal: {e}")
            return False
    
    async def update_portfolio(self, market_data: Dict[str, Any]):
        """Update portfolio positions with current market data"""
        for symbol, position in self.portfolio.items():
            if symbol in market_data:
                current_price = market_data[symbol]['price']
                position.current_price = current_price
                
                # Calculate P&L
                if position.quantity > 0:  # Long position
                    position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                else:  # Short position
                    position.unrealized_pnl = (position.entry_price - current_price) * abs(position.quantity)
                
                # Check for stop loss or take profit
                await self.check_exit_conditions(position)
    
    async def check_exit_conditions(self, position: PortfolioPosition):
        """Check if position should be closed"""
        for signal in self.active_signals:
            if signal.symbol == position.symbol:
                current_price = position.current_price
                
                # Check stop loss
                if ((signal.signal_type == "BUY" and current_price <= signal.stop_loss) or
                    (signal.signal_type == "SELL" and current_price >= signal.stop_loss)):
                    
                    await self.close_position(position, "STOP_LOSS")
                    break
                
                # Check take profit
                elif ((signal.signal_type == "BUY" and current_price >= signal.take_profit) or
                      (signal.signal_type == "SELL" and current_price <= signal.take_profit)):
                    
                    await self.close_position(position, "TAKE_PROFIT")
                    break
    
    async def close_position(self, position: PortfolioPosition, reason: str):
        """Close a position"""
        self.logger.info(f"🔒 Closing position {position.symbol}: {reason}, P&L: {position.unrealized_pnl:.2f}")
        
        # Remove from portfolio
        if position.symbol in self.portfolio:
            del self.portfolio[position.symbol]
        
        # Remove active signal
        self.active_signals = [s for s in self.active_signals if s.symbol != position.symbol]
        
        # Update performance metrics
        self.performance_metrics['total_pnl'] = self.performance_metrics.get('total_pnl', 0) + position.unrealized_pnl
        self.performance_metrics['trades_closed'] = self.performance_metrics.get('trades_closed', 0) + 1
        
        # Notify agents
        await self.agents['theo'].notify_position_closed(position, reason)
    
    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        total_pnl = sum(pos.unrealized_pnl for pos in self.portfolio.values())
        total_pnl += self.performance_metrics.get('total_pnl', 0)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_pnl": total_pnl,
            "open_positions": len(self.portfolio),
            "active_signals": len(self.active_signals),
            "trades_closed": self.performance_metrics.get('trades_closed', 0),
            "ocaml_engine_status": "running" if self.ocaml_process and self.ocaml_process.poll() is None else "stopped",
            "portfolio_details": {
                symbol: {
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "runner_mode": pos.runner_mode
                }
                for symbol, pos in self.portfolio.items()
            }
        }
        
        return report
    
    async def main_trading_loop(self):
        """Main trading loop"""
        self.logger.info("🚀 Starting Talos Capital trading loop...")
        self.running = True
        
        # Start OCaml engine
        await self.start_ocaml_signalnet()
        
        # Trading symbols
        symbols = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA"]
        
        while self.running:
            try:
                # Get market data
                market_data = await self.get_market_data(symbols)
                
                # Process signals from all sources
                ocaml_signals = await self.process_ocaml_signals()
                stoll_signals = await self.generate_stoll_ai_signals(market_data)
                
                # Combine and prioritize signals
                all_signals = ocaml_signals + stoll_signals
                
                # Execute approved signals
                for signal in all_signals:
                    await self.execute_signal(signal)
                
                # Update portfolio
                await self.update_portfolio(market_data)
                
                # Generate periodic reports
                if datetime.now().minute % 5 == 0:  # Every 5 minutes
                    report = await self.generate_performance_report()
                    await self.agents['theo'].receive_performance_report(report)
                
                # Sleep for next iteration
                await asyncio.sleep(1)  # 1 second intervals
                
            except KeyboardInterrupt:
                self.logger.info("🛑 Received shutdown signal")
                break
            except Exception as e:
                self.logger.error(f"❌ Error in main loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
        
        await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown"""
        self.logger.info("🔄 Shutting down Talos Master...")
        self.running = False
        
        # Close all positions
        for position in list(self.portfolio.values()):
            await self.close_position(position, "SHUTDOWN")
        
        # Stop OCaml process
        if self.ocaml_process:
            self.ocaml_process.terminate()
            self.ocaml_process.wait()
        
        # Final report
        final_report = await self.generate_performance_report()
        await self.agents['theo'].receive_final_report(final_report)
        
        self.logger.info("✅ Talos Master shutdown complete")

async def main():
    """Entry point"""
    orchestrator = TalosMasterOrchestrator()
    
    try:
        await orchestrator.main_trading_loop()
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested...")
    finally:
        await orchestrator.shutdown()

if __name__ == "__main__":
    print("🏛️🏛️🏛️ TALOS CAPITAL MASTER ORCHESTRATOR 🏛️🏛️🏛️")
    print("=" * 60)
    print("🧬 Multi-Agent Evolutionary Trading Empire")
    print("🤖 OCaml + Python + AI Integration")
    print("📈 Real-time Signal Processing & Execution")
    print("=" * 60)
    
    asyncio.run(main())
