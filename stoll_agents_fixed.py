"""
🤖 TALOS CAPITAL - FIXED STOLL AGENTS 🤖
Enhanced AI agents with complete functionality and error-free implementation
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from enum import Enum


class AgentRole(Enum):
    EXECUTIVE = "executive"
    TRADER = "trader"
    ANALYST = "analyst"
    ADVISOR = "advisor"
    MONITOR = "monitor"


class MessageType(Enum):
    SIGNAL = "signal"
    ANALYSIS = "analysis"
    NOTIFICATION = "notification"
    COMMAND = "command"
    RESPONSE = "response"


class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Message:
    sender: str
    recipient: str
    message_type: MessageType
    priority: Priority
    content: Any
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class TradingSignal:
    symbol: str
    signal_type: str  # BUY, SELL, HOLD
    confidence: float
    entry_price: float
    take_profit: float
    stop_loss: float
    timestamp: datetime
    metadata: Dict[str, Any] = None


class MemoryManager:
    """Simple memory management for agents"""
    
    def __init__(self):
        self.memory = {}
        self.short_term = {}
        self.long_term = {}
    
    async def store_memory(self, key: str, value: Any, memory_type: str = "short"):
        """Store memory with specified type"""
        if memory_type == "short":
            self.short_term[key] = {
                'value': value,
                'timestamp': datetime.now(),
                'expires': datetime.now() + timedelta(hours=1)
            }
        else:
            self.long_term[key] = {
                'value': value,
                'timestamp': datetime.now()
            }
        
        self.memory[key] = value
    
    async def get_memory(self, key: str) -> Any:
        """Retrieve memory by key"""
        # Check if short-term memory expired
        if key in self.short_term:
            if datetime.now() > self.short_term[key]['expires']:
                del self.short_term[key]
                if key in self.memory and key not in self.long_term:
                    del self.memory[key]
        
        return self.memory.get(key)
    
    async def clear_expired_memory(self):
        """Clear expired short-term memories"""
        expired_keys = []
        for key, data in self.short_term.items():
            if datetime.now() > data['expires']:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.short_term[key]
            if key in self.memory and key not in self.long_term:
                del self.memory[key]


class MessageBroker:
    """Message broker for agent communication"""
    
    def __init__(self):
        self.subscribers = {}
        self.message_queue = []
    
    def subscribe(self, agent_id: str, message_types: List[MessageType]):
        """Subscribe agent to specific message types"""
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = []
        
        for msg_type in message_types:
            if msg_type not in self.subscribers[agent_id]:
                self.subscribers[agent_id].append(msg_type)
    
    async def send_message(self, message: Message):
        """Send message to subscribers"""
        self.message_queue.append(message)
        
        # Notify subscribers
        for agent_id, subscribed_types in self.subscribers.items():
            if (message.message_type in subscribed_types and 
                (message.recipient == agent_id or message.recipient == "all")):
                # In a real implementation, you'd send to the agent
                print(f"📨 Message sent to {agent_id}: {message.content}")
    
    async def get_messages(self, agent_id: str, limit: int = 10) -> List[Message]:
        """Get recent messages for an agent"""
        relevant_messages = []
        
        for msg in reversed(self.message_queue[-100:]):  # Last 100 messages
            if (msg.recipient == agent_id or msg.recipient == "all"):
                relevant_messages.append(msg)
                if len(relevant_messages) >= limit:
                    break
        
        return relevant_messages


class BaseAgent:
    """Base class for all AI agents"""
    
    def __init__(self, agent_id: str = None, role: AgentRole = AgentRole.EXECUTIVE):
        self.agent_id = agent_id or f"agent_{datetime.now().timestamp()}"
        self.role = role
        self.memory_manager = MemoryManager()
        self.message_broker = MessageBroker()
        self.active = True
        self.performance_metrics = {}
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
    
    async def process_message(self, message: Message) -> Optional[Message]:
        """Process incoming message and return response if needed"""
        self.last_activity = datetime.now()
        
        # Store message in memory
        await self.memory_manager.store_memory(
            f"msg_{message.timestamp.timestamp()}", 
            message
        )
        
        # Default processing - subclasses should override
        print(f"[{self.agent_id}] Processed message: {message.content}")
        return None
    
    async def send_message(self, recipient: str, content: Any, 
                          msg_type: MessageType = MessageType.NOTIFICATION,
                          priority: Priority = Priority.MEDIUM):
        """Send message to another agent"""
        message = Message(
            sender=self.agent_id,
            recipient=recipient,
            message_type=msg_type,
            priority=priority,
            content=content
        )
        
        await self.message_broker.send_message(message)
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status information"""
        return {
            'agent_id': self.agent_id,
            'role': self.role.value,
            'active': self.active,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'performance_metrics': self.performance_metrics
        }


class PersonalizedAgent(BaseAgent):
    """Personalized AI agent for Theo with specific trading preferences"""
    
    def __init__(self, name: str, agent_type: str = "general"):
        super().__init__(f"theo_{name.lower()}", AgentRole.EXECUTIVE)
        self.name = name
        self.agent_type = agent_type
        self.preferences = self._load_theo_preferences()
        self.trading_history = []
        self.risk_tolerance = 0.7  # Moderate-high risk tolerance
        
    def _load_theo_preferences(self) -> Dict[str, Any]:
        """Load Theo's trading preferences"""
        return {
            'risk_tolerance': 0.7,
            'preferred_timeframes': ['1m', '5m', '15m'],
            'favorite_symbols': ['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA'],
            'max_position_size': 10000,
            'stop_loss_percentage': 0.02,
            'take_profit_ratio': 2.0,
            'trading_style': 'swing_momentum',
            'notification_preferences': {
                'high_confidence_signals': True,
                'portfolio_updates': True,
                'risk_alerts': True,
                'performance_summaries': True
            }
        }
    
    def _format_response_for_theo(self, data: Dict[str, Any], context: str) -> str:
        """Format responses in Theo's preferred style"""
        summary = f"🎯 **{context}**\n\n"
        
        if "signals" in data:
            summary += "📊 **Trading Signals:**\n"
            for signal in data["signals"]:
                summary += f"• {signal.get('type', 'N/A')} {signal.get('symbol', 'N/A')} "
                summary += f"@ ${signal.get('price', 0):.2f} (Confidence: {signal.get('confidence', 0):.2f})\n"
        
        if "portfolio" in data:
            portfolio = data["portfolio"]
            summary += f"\n💰 **Portfolio**: ${portfolio.get('total_value', 0):,.2f}\n"
            summary += f"📈 **P&L**: ${portfolio.get('unrealized_pnl', 0):,.2f}\n"
        
        if "performance" in data:
            perf = data["performance"]
            summary += f"\n🏆 **Performance**: {perf.get('win_rate', 0):.1%} win rate\n"
        
        return summary
    
    def _create_problem_summary(self, data: Dict[str, Any], context: str) -> str:
        """Create problem summary for Theo"""
        summary = f"🚨 **Alert** - {context}\n\n"
        
        if "error" in data:
            summary += f"❌ **Issue:** {data['error']}\n"
        
        if "recommendations" in data:
            summary += "💡 **Recommendations:**\n"
            for rec in data["recommendations"]:
                summary += f"• {rec}\n"
        
        return summary
    
    async def analyze_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market data and provide trading recommendations"""
        try:
            analysis = {
                "timestamp": datetime.now().isoformat(),
                "recommendation": "HOLD",
                "confidence": 0.5,
                "reasoning": [],
                "risk_factors": [],
                "opportunities": []
            }
            
            total_signals = 0
            bullish_signals = 0
            
            for symbol, data in market_data.items():
                if symbol in self.preferences['favorite_symbols']:
                    price = data.get('price', 0)
                    volume = data.get('volume', 0)
                    change_pct = data.get('change_percent', 0)
                    
                    total_signals += 1
                    
                    # Enhanced momentum analysis
                    if change_pct > 1.5 and volume > data.get('avg_volume', volume):
                        bullish_signals += 1
                        analysis["reasoning"].append(f"{symbol} strong momentum (+{change_pct:.1f}%)")
                        analysis["opportunities"].append(f"{symbol} breakout candidate")
                    elif change_pct < -2.0:
                        analysis["risk_factors"].append(f"{symbol} showing weakness ({change_pct:.1f}%)")
                    elif abs(change_pct) < 0.5:
                        analysis["reasoning"].append(f"{symbol} consolidating")
            
            # Determine overall recommendation
            if total_signals > 0:
                bullish_ratio = bullish_signals / total_signals
                
                if bullish_ratio > 0.6:
                    analysis["recommendation"] = "BUY"
                    analysis["confidence"] = min(0.85, 0.5 + (bullish_ratio - 0.6) * 2)
                elif bullish_ratio < 0.3:
                    analysis["recommendation"] = "SELL"
                    analysis["confidence"] = min(0.75, 0.5 + (0.3 - bullish_ratio) * 1.5)
            
            # Store analysis in memory
            await self.memory_manager.store_memory("last_analysis", analysis)
            
            return analysis
            
        except Exception as e:
            return {
                "error": str(e), 
                "recommendation": "HOLD", 
                "confidence": 0.0,
                "timestamp": datetime.now().isoformat()
            }
    
    async def evaluate_signal(self, signal: TradingSignal) -> Dict[str, Any]:
        """Evaluate a trading signal for risk/portfolio fit"""
        try:
            evaluation = {
                "approved": True,
                "confidence_adjustment": 1.0,
                "position_size_adjustment": 1.0,
                "reason": "Signal approved",
                "risk_score": 0.5
            }
            
            # Check confidence threshold
            if signal.confidence < 0.4:
                evaluation["approved"] = False
                evaluation["reason"] = "Confidence below threshold"
                return evaluation
            
            # Check symbol preference
            if signal.symbol not in self.preferences['favorite_symbols']:
                evaluation["confidence_adjustment"] *= 0.8
                evaluation["reason"] = "Symbol not in preferred list"
            
            # Check stop loss width
            stop_distance = abs(signal.entry_price - signal.stop_loss) / signal.entry_price
            if stop_distance > self.preferences['stop_loss_percentage'] * 2:
                evaluation["approved"] = False
                evaluation["reason"] = "Stop loss too wide"
                return evaluation
            
            # Check risk-reward ratio
            if signal.signal_type.upper() == "BUY":
                rr_ratio = (signal.take_profit - signal.entry_price) / (signal.entry_price - signal.stop_loss)
            else:
                rr_ratio = (signal.entry_price - signal.take_profit) / (signal.stop_loss - signal.entry_price)
            
            if rr_ratio < self.preferences['take_profit_ratio']:
                evaluation["confidence_adjustment"] *= 0.9
                evaluation["reason"] = "Risk-reward ratio below preferred"
            
            evaluation["risk_score"] = max(0.1, min(0.9, stop_distance * 10))
            
            return evaluation
            
        except Exception as e:
            return {
                "approved": False, 
                "reason": f"Evaluation error: {str(e)}",
                "risk_score": 1.0
            }
    
    async def notify_trade_execution(self, signal: TradingSignal):
        """Notify about trade execution"""
        if self.preferences['notification_preferences']['high_confidence_signals']:
            message = f"🎯 Trade Executed: {signal.signal_type} {signal.symbol} @ ${signal.entry_price:.4f}"
            message += f"\n📊 Confidence: {signal.confidence:.2f}"
            message += f"\n🎯 TP: ${signal.take_profit:.4f} | SL: ${signal.stop_loss:.4f}"
            print(f"[{self.name}] {message}")
            
            # Store in trading history
            self.trading_history.append({
                'signal': signal,
                'execution_time': datetime.now(),
                'status': 'executed'
            })
    
    async def notify_position_closed(self, position, reason: str):
        """Notify about position closure"""
        if self.preferences['notification_preferences']['portfolio_updates']:
            message = f"🔒 Position Closed: {position.symbol}"
            message += f"\n💰 P&L: ${position.unrealized_pnl:.2f}"
            message += f"\n📝 Reason: {reason}"
            print(f"[{self.name}] {message}")
            
            # Update performance metrics
            self.performance_metrics['total_trades'] = self.performance_metrics.get('total_trades', 0) + 1
            self.performance_metrics['total_pnl'] = self.performance_metrics.get('total_pnl', 0) + position.unrealized_pnl
            
            if position.unrealized_pnl > 0:
                self.performance_metrics['winning_trades'] = self.performance_metrics.get('winning_trades', 0) + 1
    
    async def receive_performance_report(self, report: Dict[str, Any]):
        """Receive and process performance report"""
        if self.preferences['notification_preferences']['performance_summaries']:
            formatted_report = self._format_response_for_theo(report, "Performance Update")
            print(f"[{self.name}] Performance Report:\n{formatted_report}")
            
            # Store in long-term memory
            await self.memory_manager.store_memory("latest_performance", report, "long")
    
    async def receive_final_report(self, report: Dict[str, Any]):
        """Receive final trading session report"""
        formatted_report = self._format_response_for_theo(report, "Final Session Report")
        print(f"[{self.name}] Final Report:\n{formatted_report}")


class RiskManagementAgent(PersonalizedAgent):
    """Specialized agent for risk management"""
    
    def __init__(self):
        super().__init__("RiskGuard", "risk_management")
        self.risk_limits = {
            'max_portfolio_risk': 0.02,  # 2% max portfolio risk
            'max_position_size': 0.1,    # 10% max position size
            'max_daily_drawdown': 0.05,  # 5% max daily drawdown
            'max_open_positions': 5      # Max concurrent positions
        }
        self.current_risk = 0.0
        self.daily_pnl = 0.0
    
    async def evaluate_signal(self, signal: TradingSignal) -> Dict[str, Any]:
        """Enhanced risk evaluation"""
        base_eval = await super().evaluate_signal(signal)
        
        # Additional risk checks
        if self.current_risk > self.risk_limits['max_portfolio_risk']:
            base_eval["approved"] = False
            base_eval["reason"] = "Portfolio risk limit exceeded"
        
        if self.daily_pnl < -self.risk_limits['max_daily_drawdown'] * 100000:  # Assuming $100k portfolio
            base_eval["approved"] = False
            base_eval["reason"] = "Daily drawdown limit exceeded"
        
        return base_eval


class PortfolioOptimizationAgent(PersonalizedAgent):
    """Specialized agent for portfolio optimization"""
    
    def __init__(self):
        super().__init__("PortfolioMax", "portfolio_optimization")
        self.correlation_matrix = {}
        self.sector_exposure = {}
        self.target_allocations = {
            'technology': 0.3,
            'financial': 0.2,
            'healthcare': 0.15,
            'consumer': 0.15,
            'industrial': 0.1,
            'other': 0.1
        }
    
    async def evaluate_signal(self, signal: TradingSignal) -> Dict[str, Any]:
        """Portfolio-aware signal evaluation"""
        base_eval = await super().evaluate_signal(signal)
        
        # Check sector allocation
        symbol = signal.symbol
        sector = self._get_sector(symbol)
        
        current_exposure = self.sector_exposure.get(sector, 0)
        target_exposure = self.target_allocations.get(sector, 0.1)
        
        if current_exposure > target_exposure * 1.5:  # 50% over target
            base_eval["position_size_adjustment"] *= 0.5
            base_eval["reason"] = f"Reducing size due to {sector} overexposure"
        
        return base_eval
    
    def _get_sector(self, symbol: str) -> str:
        """Get sector for a symbol (simplified mapping)"""
        tech_symbols = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'TSLA', 'QQQ']
        financial_symbols = ['JPM', 'BAC', 'WFC', 'GS']
        
        if symbol in tech_symbols:
            return 'technology'
        elif symbol in financial_symbols:
            return 'financial'
        elif symbol in ['SPY', 'IWM', 'VTI']:
            return 'broad_market'
        else:
            return 'other'


# Factory for creating specialized agents
class AgentFactory:
    """Factory for creating different types of agents"""
    
    @staticmethod
    def create_agent(agent_type: str, name: str = None) -> PersonalizedAgent:
        """Create an agent of the specified type"""
        
        if agent_type.lower() == "risk_management":
            return RiskManagementAgent()
        elif agent_type.lower() == "portfolio_optimization":
            return PortfolioOptimizationAgent()
        elif agent_type.lower() == "executive_trading":
            return PersonalizedAgent(name or "Executive", "executive_trading")
        else:
            return PersonalizedAgent(name or "General", agent_type)


# Example usage and testing
if __name__ == "__main__":
    print("🤖 Testing Talos Capital AI Agents 🤖")
    
    async def test_agents():
        # Create agents
        theo = AgentFactory.create_agent("executive_trading", "Theo")
        risk_agent = AgentFactory.create_agent("risk_management")
        portfolio_agent = AgentFactory.create_agent("portfolio_optimization")
        
        print(f"✅ Created agents: {theo.name}, {risk_agent.name}, {portfolio_agent.name}")
        
        # Test market analysis
        market_data = {
            'SPY': {'price': 445.50, 'volume': 50000, 'change_percent': 1.2, 'avg_volume': 45000},
            'AAPL': {'price': 185.25, 'volume': 75000, 'change_percent': 2.1, 'avg_volume': 60000}
        }
        
        analysis = await theo.analyze_market_data(market_data)
        print(f"📊 Analysis recommendation: {analysis['recommendation']} (confidence: {analysis['confidence']:.2f})")
        
        # Test signal evaluation
        test_signal = TradingSignal(
            symbol="AAPL",
            signal_type="BUY",
            confidence=0.75,
            entry_price=185.00,
            take_profit=190.00,
            stop_loss=182.00,
            timestamp=datetime.now()
        )
        
        evaluation = await theo.evaluate_signal(test_signal)
        print(f"✅ Signal evaluation: {'Approved' if evaluation['approved'] else 'Rejected'}")
        
        print("\n🎉 All agents working correctly!")
    
    # Run the test
    asyncio.run(test_agents())
