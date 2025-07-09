#!/usr/bin/env python3
"""
Stoll AI System Launcher
========================

Main entry point for the Stoll AI multi-agent system.
This script initializes all agents and provides a command-line interface
for interacting with your digital household executive system.
"""

import asyncio
import json
import logging
import sys
import signal
from datetime import datetime
from typing import Dict, Any, List
import argparse
from pathlib import Path

# Import Stoll AI components
from stoll_ai import StollAI, MemoryManager, MessageBroker, MessageType, Priority
from stoll_agents import AgentFactory
from stoll_config import get_config, validate_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stoll_ai.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class StollAILauncher:
    """Main launcher for Stoll AI system"""
    
    def __init__(self, config_env: str = "development"):
        self.config = get_config(config_env)
        validate_config(self.config)
        
        self.stoll_ai = None
        self.running = False
        self.command_history = []
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
        if self.stoll_ai:
            asyncio.create_task(self.stoll_ai.stop())
    
    async def initialize(self):
        """Initialize the Stoll AI system"""
        logger.info("🧠 Initializing Stoll AI - Your Digital Executive Assistant")
        logger.info("=" * 70)
        
        # Create data directory
        data_dir = Path(self.config["system"]["data_dir"])
        data_dir.mkdir(exist_ok=True)
        
        # Initialize Stoll AI
        self.stoll_ai = StollAI(self.config)
        
        # Add specialized agents
        self._add_specialized_agents()
        
        # Start the system
        await self.stoll_ai.start()
        
        logger.info("✅ Stoll AI system initialized successfully!")
        logger.info(f"   Environment: {self.config['system']['environment']}")
        logger.info(f"   Active Agents: {len(self.stoll_ai.agents)}")
        logger.info(f"   Data Directory: {data_dir.absolute()}")
        
        self.running = True
        
        # Run initial system check
        await self._run_initial_system_check()
    
    def _add_specialized_agents(self):
        """Add all specialized agents to the system"""
        specialized_agents = ["saas_cto", "cfo", "coo", "security", "research"]
        
        for agent_type in specialized_agents:
            try:
                agent = AgentFactory.create_agent(
                    agent_type, 
                    self.stoll_ai.memory_manager, 
                    self.stoll_ai.message_broker
                )
                self.stoll_ai.agents[agent_type] = agent
                logger.info(f"   Added {agent_type} agent: {agent.agent_id}")
            except Exception as e:
                logger.error(f"   Failed to create {agent_type} agent: {e}")
    
    async def _run_initial_system_check(self):
        """Run initial system health check"""
        logger.info("\n🔍 Running initial system check...")
        
        # Check system status
        status = self.stoll_ai.get_system_status()
        logger.info(f"   System Status: {'✅ Healthy' if status['is_running'] else '❌ Issues'}")
        
        # Test agent communication
        test_commands = [
            ("trading_cto", {"type": "analyze_market", "symbol": "SPY"}),
            ("cfo", {"type": "generate_pnl_report", "period": "daily"}),
            ("saas_cto", {"type": "monitor_health"}),
            ("research", {"type": "market_research", "symbol": "AAPL"})
        ]
        
        for agent_id, command in test_commands:
            if agent_id in self.stoll_ai.agents:
                try:
                    result = await self.stoll_ai.send_command(agent_id, command)
                    status = "✅" if result.get("status") == "success" else "⚠️"
                    logger.info(f"   {agent_id}: {status}")
                except Exception as e:
                    logger.error(f"   {agent_id}: ❌ {e}")
        
        logger.info("   Initial system check completed.")
    
    async def run_interactive_mode(self):
        """Run interactive command-line interface"""
        logger.info("\n🚀 Stoll AI Interactive Mode")
        logger.info("   Type 'help' for available commands, 'quit' to exit")
        logger.info("=" * 50)
        
        while self.running:
            try:
                # Get user input
                command = input("\nStoll AI> ").strip()
                
                if not command:
                    continue
                
                # Process command
                await self._process_command(command)
                
                # Store command history
                self.command_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "command": command
                })
                
            except EOFError:
                break
            except KeyboardInterrupt:
                print("\nUse 'quit' to exit gracefully.")
                continue
            except Exception as e:
                logger.error(f"Error processing command: {e}")
    
    async def _process_command(self, command: str):
        """Process user command"""
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == "help":
            self._show_help()
        elif cmd == "quit" or cmd == "exit":
            await self._shutdown()
        elif cmd == "status":
            await self._show_status()
        elif cmd == "agents":
            self._show_agents()
        elif cmd == "command":
            await self._send_agent_command(parts[1:])
        elif cmd == "demo":
            await self._run_demo(parts[1:])
        elif cmd == "config":
            self._show_config()
        elif cmd == "logs":
            self._show_recent_logs()
        elif cmd == "performance":
            await self._show_performance_metrics()
        else:
            print(f"Unknown command: {cmd}. Type 'help' for available commands.")
    
    def _show_help(self):
        """Show available commands"""
        help_text = """
Available Commands:
==================

System Commands:
  status          - Show system status
  agents          - List all agents
  config          - Show configuration
  logs            - Show recent logs
  performance     - Show performance metrics
  quit/exit       - Shutdown system

Agent Commands:
  command <agent> <task>  - Send command to specific agent
  
Demo Commands:
  demo trading    - Run trading demo
  demo finance    - Run finance demo
  demo operations - Run operations demo
  demo full       - Run full system demo

Examples:
  command trading_cto '{"type": "analyze_market", "symbol": "AAPL"}'
  command cfo '{"type": "generate_pnl_report"}'
  demo trading
"""
        print(help_text)
    
    async def _show_status(self):
        """Show system status"""
        status = self.stoll_ai.get_system_status()
        
        print("\n📊 System Status")
        print("=" * 30)
        print(f"Running: {'✅ Yes' if status['is_running'] else '❌ No'}")
        print(f"Agents: {len(status['agents'])}")
        print(f"Message Broker: {'✅ Active' if status['message_broker']['running'] else '❌ Inactive'}")
        print(f"Subscribers: {status['message_broker']['subscribers']}")
        
        print("\n🤖 Agent Status:")
        for agent_id, agent_status in status['agents'].items():
            running_status = "✅" if agent_status['is_running'] else "❌"
            print(f"  {agent_id}: {running_status} ({agent_status['role']})")
    
    def _show_agents(self):
        """Show all available agents"""
        print("\n🤖 Available Agents")
        print("=" * 30)
        
        for agent_id, agent in self.stoll_ai.agents.items():
            print(f"  {agent_id}: {agent.role.value}")
            print(f"    Status: {'✅ Running' if agent.is_running else '❌ Stopped'}")
            print(f"    Last Heartbeat: {agent.last_heartbeat}")
            print()
    
    async def _send_agent_command(self, args: List[str]):
        """Send command to specific agent"""
        if len(args) < 2:
            print("Usage: command <agent_id> <json_command>")
            return
        
        agent_id = args[0]
        try:
            command_json = " ".join(args[1:])
            command = json.loads(command_json)
            
            result = await self.stoll_ai.send_command(agent_id, command)
            print(f"\n📤 Command sent to {agent_id}")
            print(f"Result: {json.dumps(result, indent=2)}")
            
        except json.JSONDecodeError:
            print("Error: Invalid JSON format for command")
        except Exception as e:
            print(f"Error sending command: {e}")
    
    async def _run_demo(self, args: List[str]):
        """Run demonstration scenarios"""
        if not args:
            await self._run_full_demo()
            return
        
        demo_type = args[0].lower()
        
        if demo_type == "trading":
            await self._run_trading_demo()
        elif demo_type == "finance":
            await self._run_finance_demo()
        elif demo_type == "operations":
            await self._run_operations_demo()
        elif demo_type == "full":
            await self._run_full_demo()
        else:
            print(f"Unknown demo type: {demo_type}")
    
    async def _run_trading_demo(self):
        """Run trading demonstration"""
        print("\n🔥 Trading Demo - Stoll AI in Action")
        print("=" * 40)
        
        # Market analysis
        print("📈 Analyzing market conditions...")
        result = await self.stoll_ai.send_command("trading_cto", {
            "type": "analyze_market",
            "symbol": "SPY",
            "timeframe": "1h"
        })
        print(f"   Market Analysis: {result.get('status', 'unknown')}")
        
        # Risk assessment
        print("⚠️  Assessing portfolio risk...")
        result = await self.stoll_ai.send_command("cfo", {
            "type": "risk_assessment"
        })
        print(f"   Risk Assessment: {result.get('status', 'unknown')}")
        
        # Trading decision
        print("🎯 Making trading decision...")
        await asyncio.sleep(1)  # Simulate processing time
        print("   Decision: LONG SPY at $425.50 (Confidence: 78%)")
        print("   Stop Loss: $420.00 | Take Profit: $432.00")
        
        print("\n✅ Trading demo completed!")
    
    async def _run_finance_demo(self):
        """Run finance demonstration"""
        print("\n💰 Finance Demo - Portfolio Management")
        print("=" * 40)
        
        # Generate P&L report
        print("📊 Generating P&L report...")
        result = await self.stoll_ai.send_command("cfo", {
            "type": "generate_pnl_report",
            "period": "daily"
        })
        print(f"   P&L Report: {result.get('status', 'unknown')}")
        
        # Portfolio analysis
        print("🔍 Analyzing portfolio...")
        result = await self.stoll_ai.send_command("cfo", {
            "type": "analyze_portfolio"
        })
        print(f"   Portfolio Analysis: {result.get('status', 'unknown')}")
        
        print("\n✅ Finance demo completed!")
    
    async def _run_operations_demo(self):
        """Run operations demonstration"""
        print("\n🏠 Operations Demo - Personal Management")
        print("=" * 40)
        
        # Schedule management
        print("📅 Managing schedule...")
        result = await self.stoll_ai.send_command("coo", {
            "type": "schedule_management",
            "action": "view"
        })
        print(f"   Schedule Management: {result.get('status', 'unknown')}")
        
        # Home automation
        print("🏡 Checking home systems...")
        result = await self.stoll_ai.send_command("coo", {
            "type": "home_management",
            "system": "all"
        })
        print(f"   Home Systems: {result.get('status', 'unknown')}")
        
        print("\n✅ Operations demo completed!")
    
    async def _run_full_demo(self):
        """Run comprehensive system demonstration"""
        print("\n🌟 Full System Demo - Stoll AI Executive Suite")
        print("=" * 50)
        
        # CEO orchestration
        print("👔 CEO: Initiating system-wide analysis...")
        
        # Run all demos
        await self._run_trading_demo()
        await asyncio.sleep(1)
        await self._run_finance_demo()
        await asyncio.sleep(1)
        await self._run_operations_demo()
        
        # System health check
        print("\n🔍 Security: Running system security scan...")
        result = await self.stoll_ai.send_command("security", {
            "type": "security_scan",
            "scan_type": "quick"
        })
        print(f"   Security Scan: {result.get('status', 'unknown')}")
        
        # Research analysis
        print("📚 Research: Conducting market research...")
        result = await self.stoll_ai.send_command("research", {
            "type": "market_research",
            "symbol": "TSLA"
        })
        print(f"   Market Research: {result.get('status', 'unknown')}")
        
        print("\n🎉 Full system demo completed!")
        print("   Your Stoll AI household executive is fully operational!")
    
    def _show_config(self):
        """Show current configuration"""
        print("\n⚙️  System Configuration")
        print("=" * 30)
        print(f"Environment: {self.config['system']['environment']}")
        print(f"Debug Mode: {self.config['system']['debug']}")
        print(f"Data Directory: {self.config['system']['data_dir']}")
        print(f"Max Agents: {self.config['system']['max_agents']}")
        print(f"Heartbeat Interval: {self.config['system']['heartbeat_interval']}s")
    
    def _show_recent_logs(self):
        """Show recent log entries"""
        print("\n📋 Recent Logs")
        print("=" * 20)
        try:
            with open('stoll_ai.log', 'r') as f:
                lines = f.readlines()
                recent_lines = lines[-10:]  # Last 10 lines
                for line in recent_lines:
                    print(f"  {line.strip()}")
        except FileNotFoundError:
            print("  No log file found")
    
    async def _show_performance_metrics(self):
        """Show system performance metrics"""
        print("\n📈 Performance Metrics")
        print("=" * 30)
        
        # Get system metrics
        import psutil
        
        print(f"CPU Usage: {psutil.cpu_percent():.1f}%")
        print(f"Memory Usage: {psutil.virtual_memory().percent:.1f}%")
        print(f"Disk Usage: {psutil.disk_usage('/').percent:.1f}%")
        
        # Agent metrics
        print("\nAgent Performance:")
        for agent_id, agent in self.stoll_ai.agents.items():
            metrics = getattr(agent, 'performance_metrics', {})
            print(f"  {agent_id}: {len(metrics)} metrics recorded")
    
    async def _shutdown(self):
        """Shutdown the system gracefully"""
        print("\n🛑 Shutting down Stoll AI system...")
        
        if self.stoll_ai:
            await self.stoll_ai.stop()
        
        self.running = False
        print("   System shutdown complete. Goodbye!")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Stoll AI - Your Digital Executive Assistant")
    parser.add_argument("--env", default="development", choices=["development", "production"],
                       help="Environment to run in")
    parser.add_argument("--demo", choices=["trading", "finance", "operations", "full"],
                       help="Run specific demo and exit")
    parser.add_argument("--command", help="Execute single command and exit")
    
    args = parser.parse_args()
    
    # Initialize launcher
    launcher = StollAILauncher(args.env)
    
    try:
        # Initialize system
        await launcher.initialize()
        
        # Handle different run modes
        if args.demo:
            await launcher._run_demo([args.demo])
        elif args.command:
            await launcher._process_command(args.command)
        else:
            # Run interactive mode
            await launcher.run_interactive_mode()
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        if launcher.stoll_ai:
            await launcher.stoll_ai.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
