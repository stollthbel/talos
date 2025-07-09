"""
Stoll AI - Executive Multi-Agent System Architecture
===================================================

A distributed AI system designed to be the executive CEO of your digital household,
controlling trading systems, SaaS platforms, portfolio management, and personal automation.

Key Components:
1. Central Executive Agent (CEO)
2. Trading Agent (CTO of Trading)
3. SaaS Agent (CTO of Products)
4. Portfolio Agent (CFO)
5. Personal Agent (COO)
6. Memory & Knowledge Management
7. Inter-Agent Communication Protocol
8. Decision Engine with Risk Management
9. Real-Time Monitoring & Alerts
10. Learning & Adaptation System

Architecture: Hierarchical multi-agent system with specialized roles
Communication: Event-driven with message queues
Memory: Persistent knowledge base with vector embeddings
Learning: Reinforcement learning with human feedback
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
from pathlib import Path
import aiohttp
import numpy as np
from abc import ABC, abstractmethod
import threading
from queue import Queue, Empty
import time
import hashlib
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CORE TYPES AND ENUMS
# ============================================================================

class AgentRole(Enum):
    CEO = "ceo"           # Central Executive Agent
    CTO_TRADING = "cto_trading"  # Trading Systems
    CTO_SAAS = "cto_saas"       # SaaS Products
    CFO = "cfo"           # Portfolio & Finance
    COO = "coo"           # Personal Operations
    ANALYST = "analyst"   # Data Analysis
    SECURITY = "security" # Security & Risk

class MessageType(Enum):
    COMMAND = "command"
    QUERY = "query"
    RESPONSE = "response"
    ALERT = "alert"
    STATUS = "status"
    HEARTBEAT = "heartbeat"
    DECISION = "decision"
    RECOMMENDATION = "recommendation"

class Priority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

@dataclass
class Message:
    id: str
    from_agent: str
    to_agent: str
    type: MessageType
    priority: Priority
    content: Dict[str, Any]
    timestamp: datetime
    requires_response: bool = False
    parent_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'from_agent': self.from_agent,
            'to_agent': self.to_agent,
            'type': self.type.value,
            'priority': self.priority.value,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'requires_response': self.requires_response,
            'parent_id': self.parent_id
        }

@dataclass
class Decision:
    id: str
    agent_id: str
    decision_type: str
    context: Dict[str, Any]
    options: List[Dict[str, Any]]
    chosen_option: Optional[Dict[str, Any]]
    confidence: float
    reasoning: str
    timestamp: datetime
    outcome: Optional[str] = None

@dataclass
class KnowledgeEntry:
    id: str
    agent_id: str
    category: str
    content: Dict[str, Any]
    importance: float
    timestamp: datetime
    tags: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None

# ============================================================================
# MEMORY AND KNOWLEDGE MANAGEMENT
# ============================================================================

class MemoryManager:
    """Centralized memory and knowledge management system"""
    
    def __init__(self, db_path: str = "stoll_memory.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for persistent memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                from_agent TEXT,
                to_agent TEXT,
                type TEXT,
                priority INTEGER,
                content TEXT,
                timestamp TEXT,
                requires_response BOOLEAN,
                parent_id TEXT
            )
        ''')
        
        # Decisions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS decisions (
                id TEXT PRIMARY KEY,
                agent_id TEXT,
                decision_type TEXT,
                context TEXT,
                options TEXT,
                chosen_option TEXT,
                confidence REAL,
                reasoning TEXT,
                timestamp TEXT,
                outcome TEXT
            )
        ''')
        
        # Knowledge table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge (
                id TEXT PRIMARY KEY,
                agent_id TEXT,
                category TEXT,
                content TEXT,
                importance REAL,
                timestamp TEXT,
                tags TEXT,
                embedding TEXT
            )
        ''')
        
        # Performance metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id TEXT PRIMARY KEY,
                agent_id TEXT,
                metric_type TEXT,
                value REAL,
                timestamp TEXT,
                context TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_message(self, message: Message):
        """Store message in persistent memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            message.id, message.from_agent, message.to_agent,
            message.type.value, message.priority.value,
            json.dumps(message.content), message.timestamp.isoformat(),
            message.requires_response, message.parent_id
        ))
        
        conn.commit()
        conn.close()
    
    def store_decision(self, decision: Decision):
        """Store decision in persistent memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            decision.id, decision.agent_id, decision.decision_type,
            json.dumps(decision.context), json.dumps(decision.options),
            json.dumps(decision.chosen_option), decision.confidence,
            decision.reasoning, decision.timestamp.isoformat(),
            decision.outcome
        ))
        
        conn.commit()
        conn.close()
    
    def store_knowledge(self, knowledge: KnowledgeEntry):
        """Store knowledge entry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO knowledge VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            knowledge.id, knowledge.agent_id, knowledge.category,
            json.dumps(knowledge.content), knowledge.importance,
            knowledge.timestamp.isoformat(), json.dumps(knowledge.tags),
            json.dumps(knowledge.embedding) if knowledge.embedding else None
        ))
        
        conn.commit()
        conn.close()
    
    def get_agent_knowledge(self, agent_id: str, category: Optional[str] = None) -> List[KnowledgeEntry]:
        """Retrieve knowledge for specific agent"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = 'SELECT * FROM knowledge WHERE agent_id = ?'
        params = [agent_id]
        
        if category:
            query += ' AND category = ?'
            params.append(category)
        
        query += ' ORDER BY importance DESC, timestamp DESC'
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_knowledge(row) for row in rows]
    
    def _row_to_knowledge(self, row) -> KnowledgeEntry:
        """Convert database row to KnowledgeEntry"""
        return KnowledgeEntry(
            id=row[0],
            agent_id=row[1],
            category=row[2],
            content=json.loads(row[3]),
            importance=row[4],
            timestamp=datetime.fromisoformat(row[5]),
            tags=json.loads(row[6]),
            embedding=json.loads(row[7]) if row[7] else None
        )

# ============================================================================
# COMMUNICATION SYSTEM
# ============================================================================

class MessageBroker:
    """Central message broker for inter-agent communication"""
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.subscribers: Dict[str, List[Callable]] = {}
        self.message_queue = Queue()
        self.running = False
        self.worker_thread = None
        
    def start(self):
        """Start message broker"""
        self.running = True
        self.worker_thread = threading.Thread(target=self._message_worker)
        self.worker_thread.start()
        logger.info("Message broker started")
    
    def stop(self):
        """Stop message broker"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()
        logger.info("Message broker stopped")
    
    def subscribe(self, agent_id: str, callback: Callable):
        """Subscribe agent to receive messages"""
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = []
        self.subscribers[agent_id].append(callback)
        logger.info(f"Agent {agent_id} subscribed to message broker")
    
    def publish(self, message: Message):
        """Publish message to broker"""
        self.message_queue.put(message)
        self.memory_manager.store_message(message)
    
    def _message_worker(self):
        """Worker thread to process messages"""
        while self.running:
            try:
                message = self.message_queue.get(timeout=1)
                self._deliver_message(message)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in message worker: {e}")
    
    def _deliver_message(self, message: Message):
        """Deliver message to target agent"""
        if message.to_agent in self.subscribers:
            for callback in self.subscribers[message.to_agent]:
                try:
                    callback(message)
                except Exception as e:
                    logger.error(f"Error delivering message to {message.to_agent}: {e}")

# ============================================================================
# BASE AGENT CLASS
# ============================================================================

class BaseAgent(ABC):
    """Base class for all Stoll AI agents"""
    
    def __init__(self, agent_id: str, role: AgentRole, memory_manager: MemoryManager, 
                 message_broker: MessageBroker):
        self.agent_id = agent_id
        self.role = role
        self.memory_manager = memory_manager
        self.message_broker = message_broker
        self.is_running = False
        self.last_heartbeat = datetime.now()
        self.performance_metrics = {}
        
        # Subscribe to messages
        self.message_broker.subscribe(agent_id, self.receive_message)
        
        logger.info(f"Agent {agent_id} ({role.value}) initialized")
    
    @abstractmethod
    async def process_message(self, message: Message) -> Optional[Message]:
        """Process incoming message - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute specific task - must be implemented by subclasses"""
        pass
    
    def receive_message(self, message: Message):
        """Receive message from broker"""
        asyncio.create_task(self._handle_message(message))
    
    async def _handle_message(self, message: Message):
        """Handle incoming message"""
        try:
            logger.info(f"Agent {self.agent_id} received message from {message.from_agent}")
            response = await self.process_message(message)
            
            if response:
                self.send_message(response)
                
        except Exception as e:
            logger.error(f"Error handling message in {self.agent_id}: {e}")
    
    def send_message(self, message: Message):
        """Send message through broker"""
        self.message_broker.publish(message)
    
    def create_message(self, to_agent: str, message_type: MessageType, 
                      content: Dict[str, Any], priority: Priority = Priority.MEDIUM,
                      requires_response: bool = False) -> Message:
        """Create a new message"""
        return Message(
            id=str(uuid.uuid4()),
            from_agent=self.agent_id,
            to_agent=to_agent,
            type=message_type,
            priority=priority,
            content=content,
            timestamp=datetime.now(),
            requires_response=requires_response
        )
    
    def store_knowledge(self, category: str, content: Dict[str, Any], 
                       importance: float, tags: List[str] = None):
        """Store knowledge in memory"""
        knowledge = KnowledgeEntry(
            id=str(uuid.uuid4()),
            agent_id=self.agent_id,
            category=category,
            content=content,
            importance=importance,
            timestamp=datetime.now(),
            tags=tags or []
        )
        self.memory_manager.store_knowledge(knowledge)
    
    def get_knowledge(self, category: Optional[str] = None) -> List[KnowledgeEntry]:
        """Retrieve knowledge from memory"""
        return self.memory_manager.get_agent_knowledge(self.agent_id, category)
    
    def record_decision(self, decision_type: str, context: Dict[str, Any],
                       options: List[Dict[str, Any]], chosen_option: Dict[str, Any],
                       confidence: float, reasoning: str) -> Decision:
        """Record a decision"""
        decision = Decision(
            id=str(uuid.uuid4()),
            agent_id=self.agent_id,
            decision_type=decision_type,
            context=context,
            options=options,
            chosen_option=chosen_option,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=datetime.now()
        )
        self.memory_manager.store_decision(decision)
        return decision
    
    def send_heartbeat(self):
        """Send heartbeat to CEO"""
        heartbeat = self.create_message(
            to_agent="ceo",
            message_type=MessageType.HEARTBEAT,
            content={
                "status": "healthy",
                "last_activity": self.last_heartbeat.isoformat(),
                "performance_metrics": self.performance_metrics
            }
        )
        self.send_message(heartbeat)
    
    async def start(self):
        """Start agent"""
        self.is_running = True
        logger.info(f"Agent {self.agent_id} started")
        
        # Start heartbeat
        asyncio.create_task(self._heartbeat_loop())
    
    async def stop(self):
        """Stop agent"""
        self.is_running = False
        logger.info(f"Agent {self.agent_id} stopped")
    
    async def _heartbeat_loop(self):
        """Heartbeat loop"""
        while self.is_running:
            self.send_heartbeat()
            await asyncio.sleep(30)  # Send heartbeat every 30 seconds

# ============================================================================
# SPECIALIZED AGENTS
# ============================================================================

class CEOAgent(BaseAgent):
    """Central Executive Agent - The brain of Stoll AI"""
    
    def __init__(self, memory_manager: MemoryManager, message_broker: MessageBroker):
        super().__init__("ceo", AgentRole.CEO, memory_manager, message_broker)
        self.subordinates = {}
        self.system_status = {}
        self.active_strategies = {}
        self.risk_limits = {
            "max_portfolio_risk": 0.02,  # 2% max portfolio risk
            "max_single_trade_risk": 0.005,  # 0.5% max single trade risk
            "max_daily_trades": 100
        }
    
    async def process_message(self, message: Message) -> Optional[Message]:
        """Process incoming message as CEO"""
        if message.type == MessageType.HEARTBEAT:
            await self._handle_heartbeat(message)
        elif message.type == MessageType.ALERT:
            await self._handle_alert(message)
        elif message.type == MessageType.DECISION:
            await self._handle_decision_request(message)
        elif message.type == MessageType.QUERY:
            return await self._handle_query(message)
        
        return None
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute CEO-level task"""
        task_type = task.get("type")
        
        if task_type == "strategic_decision":
            return await self._make_strategic_decision(task)
        elif task_type == "risk_assessment":
            return await self._assess_risk(task)
        elif task_type == "resource_allocation":
            return await self._allocate_resources(task)
        
        return {"status": "error", "message": f"Unknown task type: {task_type}"}
    
    async def _handle_heartbeat(self, message: Message):
        """Handle heartbeat from subordinate"""
        agent_id = message.from_agent
        self.system_status[agent_id] = {
            "last_heartbeat": datetime.now(),
            "status": message.content.get("status", "unknown"),
            "performance": message.content.get("performance_metrics", {})
        }
    
    async def _handle_alert(self, message: Message):
        """Handle alert from subordinate"""
        alert_type = message.content.get("type")
        severity = message.content.get("severity", "medium")
        
        if severity == "critical":
            # Immediate action required
            await self._handle_critical_alert(message)
        elif severity == "high":
            # Escalate to appropriate agent
            await self._escalate_alert(message)
        
        # Log alert
        self.store_knowledge(
            category="alerts",
            content=message.content,
            importance=1.0 if severity == "critical" else 0.7,
            tags=[alert_type, severity]
        )
    
    async def _handle_decision_request(self, message: Message):
        """Handle decision request from subordinate"""
        decision_context = message.content
        
        # Analyze decision context
        risk_assessment = await self._assess_decision_risk(decision_context)
        
        # Make decision based on risk and strategy
        decision = await self._make_executive_decision(decision_context, risk_assessment)
        
        # Send decision back
        response = self.create_message(
            to_agent=message.from_agent,
            message_type=MessageType.RESPONSE,
            content=decision,
            priority=message.priority
        )
        self.send_message(response)
    
    async def _make_strategic_decision(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Make high-level strategic decision"""
        context = task.get("context", {})
        options = task.get("options", [])
        
        # Analyze each option
        analyzed_options = []
        for option in options:
            risk_score = await self._calculate_option_risk(option)
            reward_score = await self._calculate_option_reward(option)
            
            analyzed_options.append({
                "option": option,
                "risk_score": risk_score,
                "reward_score": reward_score,
                "risk_adjusted_score": reward_score / (risk_score + 0.01)
            })
        
        # Choose best option
        best_option = max(analyzed_options, key=lambda x: x["risk_adjusted_score"])
        
        # Record decision
        decision = self.record_decision(
            decision_type="strategic",
            context=context,
            options=options,
            chosen_option=best_option["option"],
            confidence=min(best_option["risk_adjusted_score"], 1.0),
            reasoning=f"Selected option with highest risk-adjusted score: {best_option['risk_adjusted_score']:.3f}"
        )
        
        return {
            "status": "success",
            "decision": best_option["option"],
            "confidence": decision.confidence,
            "reasoning": decision.reasoning
        }
    
    async def _assess_decision_risk(self, context: Dict[str, Any]) -> float:
        """Assess risk of a decision"""
        # Implement risk assessment logic
        base_risk = 0.1
        
        # Adjust based on context
        if context.get("financial_impact", 0) > 1000:
            base_risk += 0.3
        
        if context.get("time_sensitive", False):
            base_risk += 0.2
        
        return min(base_risk, 1.0)
    
    async def _calculate_option_risk(self, option: Dict[str, Any]) -> float:
        """Calculate risk score for an option"""
        # Implement option risk calculation
        return option.get("risk_score", 0.5)
    
    async def _calculate_option_reward(self, option: Dict[str, Any]) -> float:
        """Calculate reward score for an option"""
        # Implement option reward calculation
        return option.get("reward_score", 0.5)

class TradingAgent(BaseAgent):
    """Trading CTO Agent - Manages all trading operations"""
    
    def __init__(self, memory_manager: MemoryManager, message_broker: MessageBroker):
        super().__init__("trading_cto", AgentRole.CTO_TRADING, memory_manager, message_broker)
        self.active_strategies = {}
        self.position_manager = None
        self.risk_manager = None
        self.signal_engine = None
    
    async def process_message(self, message: Message) -> Optional[Message]:
        """Process trading-related message"""
        if message.type == MessageType.COMMAND:
            return await self._handle_command(message)
        elif message.type == MessageType.QUERY:
            return await self._handle_query(message)
        
        return None
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trading task"""
        task_type = task.get("type")
        
        if task_type == "start_strategy":
            return await self._start_strategy(task)
        elif task_type == "stop_strategy":
            return await self._stop_strategy(task)
        elif task_type == "analyze_market":
            return await self._analyze_market(task)
        elif task_type == "execute_trade":
            return await self._execute_trade(task)
        
        return {"status": "error", "message": f"Unknown task type: {task_type}"}
    
    async def _handle_command(self, message: Message) -> Message:
        """Handle command from CEO"""
        command = message.content.get("command")
        
        if command == "start_trading":
            result = await self._start_trading_systems()
        elif command == "stop_trading":
            result = await self._stop_trading_systems()
        elif command == "update_strategy":
            result = await self._update_strategy(message.content.get("strategy"))
        else:
            result = {"status": "error", "message": f"Unknown command: {command}"}
        
        return self.create_message(
            to_agent=message.from_agent,
            message_type=MessageType.RESPONSE,
            content=result
        )
    
    async def _start_trading_systems(self) -> Dict[str, Any]:
        """Start all trading systems"""
        # Initialize signal engine
        # Start position monitoring
        # Connect to data feeds
        
        self.store_knowledge(
            category="system_events",
            content={"event": "trading_systems_started", "timestamp": datetime.now().isoformat()},
            importance=0.8,
            tags=["trading", "system", "startup"]
        )
        
        return {"status": "success", "message": "Trading systems started"}
    
    async def _stop_trading_systems(self) -> Dict[str, Any]:
        """Stop all trading systems"""
        # Close all positions
        # Stop data feeds
        # Save state
        
        return {"status": "success", "message": "Trading systems stopped"}
    
    async def _analyze_market(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market conditions"""
        symbol = task.get("symbol", "SPY")
        timeframe = task.get("timeframe", "1h")
        
        # Implement market analysis
        analysis = {
            "symbol": symbol,
            "timeframe": timeframe,
            "trend": "bullish",
            "strength": 0.75,
            "volatility": 0.15,
            "volume": "high",
            "signals": {
                "entry": "long",
                "confidence": 0.8,
                "stop_loss": 420.0,
                "take_profit": 430.0
            }
        }
        
        return {"status": "success", "analysis": analysis}

# ============================================================================
# STOLL AI ORCHESTRATOR
# ============================================================================

class StollAI:
    """Main orchestrator for the Stoll AI system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.memory_manager = MemoryManager()
        self.message_broker = MessageBroker(self.memory_manager)
        self.agents = {}
        self.is_running = False
        
        # Initialize agents
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all agents"""
        self.agents["ceo"] = CEOAgent(self.memory_manager, self.message_broker)
        self.agents["trading_cto"] = TradingAgent(self.memory_manager, self.message_broker)
        
        # TODO: Add more agents
        # self.agents["saas_cto"] = SaaSAgent(self.memory_manager, self.message_broker)
        # self.agents["cfo"] = CFOAgent(self.memory_manager, self.message_broker)
        # self.agents["coo"] = COOAgent(self.memory_manager, self.message_broker)
    
    async def start(self):
        """Start the Stoll AI system"""
        logger.info("Starting Stoll AI system...")
        
        # Start message broker
        self.message_broker.start()
        
        # Start all agents
        for agent in self.agents.values():
            await agent.start()
        
        self.is_running = True
        logger.info("Stoll AI system started successfully")
        
        # Send startup message
        startup_message = self.agents["ceo"].create_message(
            to_agent="all",
            message_type=MessageType.STATUS,
            content={
                "status": "system_startup",
                "timestamp": datetime.now().isoformat(),
                "message": "Stoll AI system is now online and ready for operations"
            },
            priority=Priority.HIGH
        )
        self.message_broker.publish(startup_message)
    
    async def stop(self):
        """Stop the Stoll AI system"""
        logger.info("Stopping Stoll AI system...")
        
        # Stop all agents
        for agent in self.agents.values():
            await agent.stop()
        
        # Stop message broker
        self.message_broker.stop()
        
        self.is_running = False
        logger.info("Stoll AI system stopped")
    
    async def send_command(self, agent_id: str, command: Dict[str, Any]) -> Dict[str, Any]:
        """Send command to specific agent"""
        if agent_id not in self.agents:
            return {"status": "error", "message": f"Agent {agent_id} not found"}
        
        # Create command message
        message = self.agents["ceo"].create_message(
            to_agent=agent_id,
            message_type=MessageType.COMMAND,
            content=command,
            priority=Priority.HIGH,
            requires_response=True
        )
        
        # Send message
        self.message_broker.publish(message)
        
        return {"status": "success", "message": "Command sent"}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        status = {
            "is_running": self.is_running,
            "agents": {},
            "message_broker": {
                "running": self.message_broker.running,
                "subscribers": len(self.message_broker.subscribers)
            }
        }
        
        for agent_id, agent in self.agents.items():
            status["agents"][agent_id] = {
                "role": agent.role.value,
                "is_running": agent.is_running,
                "last_heartbeat": agent.last_heartbeat.isoformat()
            }
        
        return status

# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

async def main():
    """Example usage of Stoll AI"""
    
    # Initialize Stoll AI
    stoll = StollAI()
    
    try:
        # Start the system
        await stoll.start()
        
        # Give it a moment to initialize
        await asyncio.sleep(2)
        
        # Get system status
        status = stoll.get_system_status()
        print("System Status:")
        print(json.dumps(status, indent=2))
        
        # Send a command to the trading agent
        trade_command = {
            "type": "analyze_market",
            "symbol": "SPY",
            "timeframe": "1h"
        }
        
        result = await stoll.send_command("trading_cto", trade_command)
        print(f"\nCommand result: {result}")
        
        # Let it run for a bit
        await asyncio.sleep(10)
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await stoll.stop()

if __name__ == "__main__":
    asyncio.run(main())
