#!/usr/bin/env python3
"""
🚀 TALOS CAPITAL LAUNCHER 🚀
One-click startup for the entire Talos trading ecosystem

This launcher coordinates:
- OCaml SignalNet evolutionary engine
- Python Stoll AI agents
- Master orchestrator
- Household CEO AI
- Web dashboard
- All data sources and monitoring
"""

import subprocess
import time
import sys
import os
import signal
import threading
from pathlib import Path
import psutil
import json
from datetime import datetime

class TalosLauncher:
    def __init__(self):
        self.processes = {}
        self.running = True
        self.launch_log = []
        
    def log(self, message: str):
        """Log a message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.launch_log.append(log_entry)
        print(log_entry)
    
    def check_dependencies(self):
        """Check if all required dependencies are available"""
        self.log("🔍 Checking dependencies...")
        
        # Check Python dependencies
        python_deps = ['yfinance', 'pandas', 'numpy', 'flask', 'asyncio']
        missing_deps = []
        
        for dep in python_deps:
            try:
                __import__(dep)
                self.log(f"✅ Python: {dep}")
            except ImportError:
                missing_deps.append(dep)
                self.log(f"❌ Python: {dep} - MISSING")
        
        # Check OCaml
        try:
            result = subprocess.run(['ocaml', '-version'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.log(f"✅ OCaml: {result.stdout.strip()}")
            else:
                self.log("❌ OCaml: Not found")
        except:
            self.log("❌ OCaml: Not found")
        
        # Check Node.js for frontend
        try:
            result = subprocess.run(['node', '--version'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.log(f"✅ Node.js: {result.stdout.strip()}")
            else:
                self.log("❌ Node.js: Not found")
        except:
            self.log("❌ Node.js: Not found")
        
        if missing_deps:
            self.log(f"⚠️ Missing Python dependencies: {', '.join(missing_deps)}")
            self.log("💡 Run: pip install " + " ".join(missing_deps))
    
    def compile_ocaml_engine(self):
        """Compile the OCaml SignalNet engine"""
        self.log("🧬 Compiling OCaml SignalNet engine...")
        
        ocaml_path = Path("OCaml")
        if not ocaml_path.exists():
            self.log("❌ OCaml directory not found")
            return False
        
        try:
            # Try to compile the signal evolution engine
            if (ocaml_path / "signal_evolution.ml").exists():
                result = subprocess.run(
                    ["ocamlc", "-o", "signal_evolution", "signal_evolution.ml"],
                    cwd=str(ocaml_path),
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    self.log("✅ OCaml SignalNet compiled successfully")
                    return True
                else:
                    self.log(f"❌ OCaml compilation failed: {result.stderr}")
            
            # Fallback to signalnet.ml
            elif (ocaml_path / "signalnet.ml").exists():
                result = subprocess.run(
                    ["ocamlc", "-o", "signalnet", "signalnet.ml"],
                    cwd=str(ocaml_path),
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    self.log("✅ OCaml SignalNet compiled successfully")
                    return True
                else:
                    self.log(f"❌ OCaml compilation failed: {result.stderr}")
            
            else:
                self.log("❌ No OCaml source files found")
            
        except Exception as e:
            self.log(f"❌ OCaml compilation error: {str(e)}")
        
        return False
    
    def start_ocaml_engine(self):
        """Start the OCaml evolutionary engine"""
        self.log("🧬 Starting OCaml SignalNet engine...")
        
        ocaml_path = Path("OCaml")
        
        # Try signal_evolution first, then signalnet
        executables = ["signal_evolution", "signalnet"]
        
        for exe in executables:
            exe_path = ocaml_path / exe
            if exe_path.exists():
                try:
                    process = subprocess.Popen(
                        [f"./{exe}"],
                        cwd=str(ocaml_path),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    self.processes['ocaml_engine'] = process
                    self.log(f"✅ OCaml engine started (PID: {process.pid})")
                    return True
                except Exception as e:
                    self.log(f"❌ Failed to start {exe}: {str(e)}")
        
        self.log("❌ No OCaml executable found")
        return False
    
    def start_master_orchestrator(self):
        """Start the master orchestrator"""
        self.log("🎯 Starting Master Orchestrator...")
        
        try:
            process = subprocess.Popen(
                [sys.executable, "talos_master_orchestrator.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes['orchestrator'] = process
            self.log(f"✅ Master Orchestrator started (PID: {process.pid})")
            return True
        except Exception as e:
            self.log(f"❌ Failed to start Master Orchestrator: {str(e)}")
            return False
    
    def start_household_ceo(self):
        """Start the Household CEO AI"""
        self.log("🏛️ Starting Household CEO AI...")
        
        try:
            process = subprocess.Popen(
                [sys.executable, "stoll_household_ceo.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes['household_ceo'] = process
            self.log(f"✅ Household CEO AI started (PID: {process.pid})")
            return True
        except Exception as e:
            self.log(f"❌ Failed to start Household CEO: {str(e)}")
            return False
    
    def start_web_dashboard(self):
        """Start the web dashboard"""
        self.log("🌐 Starting web dashboard...")
        
        try:
            # Start backend
            backend_process = subprocess.Popen(
                [sys.executable, "talos-backend.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes['backend'] = backend_process
            self.log(f"✅ Backend started (PID: {backend_process.pid})")
            
            # Wait a moment for backend to start
            time.sleep(2)
            
            # Start frontend server
            frontend_process = subprocess.Popen(
                [sys.executable, "-m", "http.server", "3000"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes['frontend'] = frontend_process
            self.log(f"✅ Frontend started (PID: {frontend_process.pid})")
            
            return True
        except Exception as e:
            self.log(f"❌ Failed to start web dashboard: {str(e)}")
            return False
    
    def monitor_processes(self):
        """Monitor all running processes"""
        while self.running:
            try:
                for name, process in list(self.processes.items()):
                    if process.poll() is not None:
                        self.log(f"⚠️ Process {name} stopped (exit code: {process.returncode})")
                        # Optionally restart critical processes
                        if name in ['ocaml_engine', 'orchestrator']:
                            self.log(f"🔄 Attempting to restart {name}...")
                            if name == 'ocaml_engine':
                                self.start_ocaml_engine()
                            elif name == 'orchestrator':
                                self.start_master_orchestrator()
                
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                self.log(f"❌ Monitor error: {str(e)}")
                time.sleep(10)
    
    def get_system_status(self):
        """Get current system status"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "processes": {},
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent
            }
        }
        
        for name, process in self.processes.items():
            try:
                if process.poll() is None:
                    proc = psutil.Process(process.pid)
                    status["processes"][name] = {
                        "status": "running",
                        "pid": process.pid,
                        "cpu_percent": proc.cpu_percent(),
                        "memory_percent": proc.memory_percent()
                    }
                else:
                    status["processes"][name] = {
                        "status": "stopped",
                        "exit_code": process.returncode
                    }
            except:
                status["processes"][name] = {"status": "unknown"}
        
        return status
    
    def shutdown(self):
        """Graceful shutdown of all processes"""
        self.log("🛑 Initiating graceful shutdown...")
        self.running = False
        
        for name, process in self.processes.items():
            try:
                if process.poll() is None:
                    self.log(f"🔄 Stopping {name} (PID: {process.pid})...")
                    process.terminate()
                    
                    # Wait up to 10 seconds for graceful shutdown
                    try:
                        process.wait(timeout=10)
                        self.log(f"✅ {name} stopped gracefully")
                    except subprocess.TimeoutExpired:
                        self.log(f"⚠️ Force killing {name}...")
                        process.kill()
                        process.wait()
                        self.log(f"✅ {name} force stopped")
            except Exception as e:
                self.log(f"❌ Error stopping {name}: {str(e)}")
        
        self.log("✅ All processes stopped")
    
    def launch_full_system(self):
        """Launch the complete Talos ecosystem"""
        self.log("🚀🚀🚀 TALOS CAPITAL ECOSYSTEM LAUNCH 🚀🚀🚀")
        self.log("=" * 60)
        
        # Check dependencies
        self.check_dependencies()
        
        # Compile OCaml engine
        ocaml_compiled = self.compile_ocaml_engine()
        
        # Start core components
        components_started = []
        
        if ocaml_compiled:
            if self.start_ocaml_engine():
                components_started.append("OCaml SignalNet")
        
        if self.start_master_orchestrator():
            components_started.append("Master Orchestrator")
        
        if self.start_household_ceo():
            components_started.append("Household CEO AI")
        
        if self.start_web_dashboard():
            components_started.append("Web Dashboard")
        
        # Start process monitor
        monitor_thread = threading.Thread(target=self.monitor_processes, daemon=True)
        monitor_thread.start()
        
        # Summary
        self.log("=" * 60)
        self.log(f"✅ {len(components_started)} components started successfully:")
        for component in components_started:
            self.log(f"   • {component}")
        
        if components_started:
            self.log("🌐 Access points:")
            self.log("   • Web Dashboard: http://localhost:3000")
            self.log("   • API Backend: http://localhost:5000")
            self.log("   • Household CEO: Terminal interface")
            
            self.log("💡 Commands:")
            self.log("   • Press 's' for system status")
            self.log("   • Press 'q' to quit")
            self.log("=" * 60)
            
            return True
        else:
            self.log("❌ No components started successfully")
            return False
    
    def interactive_mode(self):
        """Interactive command mode"""
        try:
            while self.running:
                command = input("\n🎮 Talos Command [s=status, q=quit]: ").strip().lower()
                
                if command == 'q' or command == 'quit':
                    break
                elif command == 's' or command == 'status':
                    status = self.get_system_status()
                    self.log("📊 SYSTEM STATUS:")
                    self.log(f"   CPU: {status['system']['cpu_percent']:.1f}%")
                    self.log(f"   Memory: {status['system']['memory_percent']:.1f}%")
                    self.log(f"   Disk: {status['system']['disk_percent']:.1f}%")
                    self.log("   Processes:")
                    for name, proc_status in status['processes'].items():
                        status_icon = "🟢" if proc_status.get('status') == 'running' else "🔴"
                        self.log(f"     {status_icon} {name}: {proc_status.get('status', 'unknown')}")
                elif command:
                    self.log("❓ Unknown command. Try 's' for status or 'q' to quit.")
        
        except KeyboardInterrupt:
            pass

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\n🛑 Shutdown signal received...")
    launcher.shutdown()
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handlers
    launcher = TalosLauncher()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Launch the full system
        if launcher.launch_full_system():
            # Enter interactive mode
            launcher.interactive_mode()
        
    except Exception as e:
        launcher.log(f"❌ Fatal error: {str(e)}")
    
    finally:
        launcher.shutdown()
        launcher.log("👋 Talos Capital Launcher shutdown complete")
