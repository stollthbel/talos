#!/usr/bin/env python3
"""
Stoll AI System Health Check Script
Comprehensive health monitoring and diagnostics for the Stoll AI system
"""

import asyncio
import json
import logging
import sys
import time
from typing import Dict, List, Optional, Tuple
import aiohttp
import redis
import sqlite3
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StollHealthChecker:
    """Comprehensive health checker for Stoll AI system"""
    
    def __init__(self):
        self.services = {
            'stoll_ai_api': 'http://localhost:8001',
            'talos_backend': 'http://localhost:5000',
            'nginx': 'http://localhost:80',
            'grafana': 'http://localhost:3000',
            'prometheus': 'http://localhost:9091'
        }
        self.redis_host = 'localhost'
        self.redis_port = 6379
        self.db_path = './data/stoll_ai.db'
        
    async def check_http_service(self, name: str, url: str, timeout: int = 10) -> Tuple[bool, str]:
        """Check if an HTTP service is healthy"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                # Try health endpoint first
                health_url = f"{url}/health"
                async with session.get(health_url) as response:
                    if response.status == 200:
                        return True, f"{name} is healthy"
                    else:
                        return False, f"{name} returned status {response.status}"
        except aiohttp.ClientError as e:
            try:
                # Try root endpoint if health endpoint fails
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                    async with session.get(url) as response:
                        if response.status in [200, 404]:  # 404 is OK for root
                            return True, f"{name} is responding (status {response.status})"
                        else:
                            return False, f"{name} returned status {response.status}"
            except Exception as e2:
                return False, f"{name} is not responding: {str(e)}"
        except Exception as e:
            return False, f"{name} check failed: {str(e)}"
    
    def check_redis(self) -> Tuple[bool, str]:
        """Check Redis connection"""
        try:
            r = redis.Redis(host=self.redis_host, port=self.redis_port, db=0)
            r.ping()
            return True, "Redis is healthy"
        except Exception as e:
            return False, f"Redis check failed: {str(e)}"
    
    def check_database(self) -> Tuple[bool, str]:
        """Check SQLite database"""
        try:
            if not Path(self.db_path).exists():
                return False, f"Database file {self.db_path} does not exist"
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            conn.close()
            
            return True, f"Database is healthy ({len(tables)} tables)"
        except Exception as e:
            return False, f"Database check failed: {str(e)}"
    
    def check_file_system(self) -> Tuple[bool, str]:
        """Check required directories and files"""
        required_paths = [
            './data',
            './logs',
            './stoll_ai.py',
            './stoll_agents.py',
            './stoll_config.py',
            './stoll_api.py',
            './stoll_launcher.py'
        ]
        
        missing_paths = []
        for path in required_paths:
            if not Path(path).exists():
                missing_paths.append(path)
        
        if missing_paths:
            return False, f"Missing required paths: {', '.join(missing_paths)}"
        else:
            return True, "All required files and directories exist"
    
    async def check_stoll_agents(self) -> Tuple[bool, str]:
        """Check if Stoll AI agents are responding"""
        try:
            async with aiohttp.ClientSession() as session:
                # Check agent status endpoint
                async with session.get('http://localhost:8001/agents/status') as response:
                    if response.status == 200:
                        data = await response.json()
                        active_agents = data.get('active_agents', 0)
                        return True, f"Stoll AI agents are healthy ({active_agents} active)"
                    else:
                        return False, f"Agent status endpoint returned {response.status}"
        except Exception as e:
            return False, f"Agent check failed: {str(e)}"
    
    async def check_system_resources(self) -> Tuple[bool, str]:
        """Check system resources"""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            warnings = []
            if cpu_percent > 80:
                warnings.append(f"High CPU usage: {cpu_percent}%")
            if memory_percent > 80:
                warnings.append(f"High memory usage: {memory_percent}%")
            if disk_percent > 80:
                warnings.append(f"High disk usage: {disk_percent}%")
            
            if warnings:
                return False, f"Resource warnings: {', '.join(warnings)}"
            else:
                return True, f"System resources OK (CPU: {cpu_percent}%, Memory: {memory_percent}%, Disk: {disk_percent}%)"
        except ImportError:
            return True, "System resource check skipped (psutil not available)"
        except Exception as e:
            return False, f"System resource check failed: {str(e)}"
    
    async def run_comprehensive_check(self) -> Dict:
        """Run all health checks"""
        results = {}
        
        # Check HTTP services
        for name, url in self.services.items():
            is_healthy, message = await self.check_http_service(name, url)
            results[name] = {'healthy': is_healthy, 'message': message}
        
        # Check Redis
        is_healthy, message = self.check_redis()
        results['redis'] = {'healthy': is_healthy, 'message': message}
        
        # Check Database
        is_healthy, message = self.check_database()
        results['database'] = {'healthy': is_healthy, 'message': message}
        
        # Check File System
        is_healthy, message = self.check_file_system()
        results['filesystem'] = {'healthy': is_healthy, 'message': message}
        
        # Check Stoll Agents
        is_healthy, message = await self.check_stoll_agents()
        results['stoll_agents'] = {'healthy': is_healthy, 'message': message}
        
        # Check System Resources
        is_healthy, message = await self.check_system_resources()
        results['system_resources'] = {'healthy': is_healthy, 'message': message}
        
        return results
    
    def print_results(self, results: Dict):
        """Print formatted health check results"""
        print("\n🏥 Stoll AI System Health Check Results")
        print("=" * 50)
        
        healthy_count = 0
        total_count = len(results)
        
        for service, result in results.items():
            status = "✅" if result['healthy'] else "❌"
            print(f"{status} {service.replace('_', ' ').title()}: {result['message']}")
            if result['healthy']:
                healthy_count += 1
        
        print("=" * 50)
        health_percentage = (healthy_count / total_count) * 100
        
        if health_percentage == 100:
            print("🎉 All systems are healthy!")
        elif health_percentage >= 80:
            print(f"⚠️  System health: {health_percentage:.1f}% ({healthy_count}/{total_count})")
        else:
            print(f"🚨 System health: {health_percentage:.1f}% ({healthy_count}/{total_count})")
            print("   Please address the issues above.")
        
        return health_percentage

async def main():
    """Main health check function"""
    checker = StollHealthChecker()
    
    print("🔍 Starting Stoll AI System Health Check...")
    start_time = time.time()
    
    try:
        results = await checker.run_comprehensive_check()
        health_percentage = checker.print_results(results)
        
        end_time = time.time()
        print(f"\n⏱️  Health check completed in {end_time - start_time:.2f} seconds")
        
        # Exit with appropriate code
        if health_percentage < 80:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        print(f"\n❌ Health check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
