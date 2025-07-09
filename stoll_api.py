"""
Stoll AI REST API
=================

FastAPI-based REST API for the Stoll AI system.
Provides external access to agent commands, queries, and system status.
"""

from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import asyncio
import jwt
import logging
from datetime import datetime, timedelta
import os

from stoll_ai import StollAI
from stoll_agents import AgentFactory
from stoll_config import get_final_config

# Initialize FastAPI app
app = FastAPI(
    title="Stoll AI API",
    description="Executive Multi-Agent System API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration
config = get_final_config()
security = HTTPBearer()

# Global Stoll AI instance
stoll_ai: Optional[StollAI] = None

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config["api"]["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class CommandRequest(BaseModel):
    agent_id: str = Field(..., description="Target agent ID")
    command: str = Field(..., description="Command to execute")
    params: Optional[Dict[str, Any]] = Field(default=None, description="Command parameters")

class QueryRequest(BaseModel):
    agent_id: str = Field(..., description="Target agent ID")
    query: str = Field(..., description="Query to execute")
    params: Optional[Dict[str, Any]] = Field(default=None, description="Query parameters")

class AgentStatus(BaseModel):
    agent_id: str
    role: str
    status: str
    last_activity: datetime
    performance_metrics: Dict[str, Any]

class SystemStatus(BaseModel):
    timestamp: datetime
    agents: int
    active_agents: int
    system_health: str
    recent_alerts: int
    uptime: str

# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify JWT token"""
    if not config["api"]["authentication"]["enabled"]:
        return {"user": "system", "role": "admin"}
    
    try:
        payload = jwt.decode(
            credentials.credentials,
            config["api"]["authentication"]["jwt_secret"],
            algorithms=["HS256"]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize Stoll AI system on startup"""
    global stoll_ai
    
    logging.info("Starting Stoll AI API...")
    
    # Initialize Stoll AI
    stoll_ai = StollAI()
    
    # Create and register all agents
    for agent_type in AgentFactory.get_available_agents():
        try:
            agent = AgentFactory.create_agent(
                agent_type,
                stoll_ai.memory_manager,
                stoll_ai.message_broker
            )
            stoll_ai.register_agent(agent)
            logging.info(f"Registered agent: {agent_type}")
        except Exception as e:
            logging.error(f"Failed to create agent {agent_type}: {e}")
    
    # Start the system
    await stoll_ai.start()
    
    logging.info("Stoll AI API started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global stoll_ai
    
    logging.info("Shutting down Stoll AI API...")
    
    if stoll_ai:
        await stoll_ai.stop()
    
    logging.info("Stoll AI API shutdown complete")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not stoll_ai or not stoll_ai.running:
        raise HTTPException(status_code=503, detail="System not ready")
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": config["system"]["version"]
    }

# System status endpoint
@app.get("/status", response_model=SystemStatus)
async def get_system_status(user: dict = Depends(verify_token)):
    """Get overall system status"""
    if not stoll_ai:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        status = await stoll_ai.get_system_status()
        return SystemStatus(**status, uptime="unknown")  # TODO: Calculate actual uptime
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {e}")

# Agent management endpoints
@app.get("/agents")
async def list_agents(user: dict = Depends(verify_token)):
    """List all registered agents"""
    if not stoll_ai:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    agents = []
    for agent_id, agent in stoll_ai.agents.items():
        agents.append({
            "agent_id": agent_id,
            "role": agent.role.value,
            "status": "active" if agent.is_active else "inactive",
            "last_activity": agent.last_activity.isoformat()
        })
    
    return {"agents": agents}

@app.get("/agents/{agent_id}/status")
async def get_agent_status(agent_id: str, user: dict = Depends(verify_token)):
    """Get status of a specific agent"""
    if not stoll_ai:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if agent_id not in stoll_ai.agents:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    agent = stoll_ai.agents[agent_id]
    
    return {
        "agent_id": agent_id,
        "role": agent.role.value,
        "status": "active" if agent.is_active else "inactive",
        "last_activity": agent.last_activity.isoformat(),
        "performance_metrics": agent.performance_metrics
    }

# Command execution endpoints
@app.post("/command")
async def execute_command(request: CommandRequest, user: dict = Depends(verify_token)):
    """Execute a command on a specific agent"""
    if not stoll_ai:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        await stoll_ai.send_command(request.agent_id, request.command, request.params)
        return {
            "status": "success",
            "message": f"Command '{request.command}' sent to agent '{request.agent_id}'",
            "timestamp": datetime.now().isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Command execution failed: {e}")

@app.post("/query")
async def execute_query(request: QueryRequest, user: dict = Depends(verify_token)):
    """Execute a query on a specific agent"""
    if not stoll_ai:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        await stoll_ai.query_agent(request.agent_id, request.query, request.params)
        return {
            "status": "success",
            "message": f"Query '{request.query}' sent to agent '{request.agent_id}'",
            "timestamp": datetime.now().isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution failed: {e}")

# Trading-specific endpoints
@app.get("/trading/signals")
async def get_trading_signals(
    symbol: str = "SPY",
    timeframe: str = "1d",
    limit: int = 100,
    user: dict = Depends(verify_token)
):
    """Get recent trading signals"""
    if not stoll_ai:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Query the research agent for signals
        await stoll_ai.query_agent("research", "get_signals", {
            "symbol": symbol,
            "timeframe": timeframe,
            "limit": limit
        })
        
        return {
            "status": "success",
            "message": "Signal request sent to research agent",
            "symbol": symbol,
            "timeframe": timeframe
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get signals: {e}")

@app.get("/portfolio/status")
async def get_portfolio_status(user: dict = Depends(verify_token)):
    """Get current portfolio status"""
    if not stoll_ai:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        await stoll_ai.query_agent("cfo", "portfolio_status")
        
        return {
            "status": "success",
            "message": "Portfolio status request sent to CFO agent"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get portfolio status: {e}")

# SaaS management endpoints
@app.get("/saas/status")
async def get_saas_status(user: dict = Depends(verify_token)):
    """Get SaaS services status"""
    if not stoll_ai:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        await stoll_ai.query_agent("saas_cto", "service_status")
        
        return {
            "status": "success",
            "message": "SaaS status request sent to SaaS CTO agent"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get SaaS status: {e}")

# Knowledge and memory endpoints
@app.get("/knowledge/{agent_id}")
async def get_agent_knowledge(
    agent_id: str,
    category: Optional[str] = None,
    tags: Optional[str] = None,
    limit: int = 50,
    user: dict = Depends(verify_token)
):
    """Get knowledge items for a specific agent"""
    if not stoll_ai:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if agent_id not in stoll_ai.agents:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    try:
        agent = stoll_ai.agents[agent_id]
        tag_list = tags.split(",") if tags else None
        
        knowledge_items = agent.get_knowledge(category, tag_list, limit)
        
        return {
            "agent_id": agent_id,
            "category": category,
            "tags": tag_list,
            "count": len(knowledge_items),
            "items": [
                {
                    "id": item.id,
                    "category": item.category,
                    "content": item.content,
                    "importance": item.importance,
                    "tags": item.tags,
                    "created_at": item.created_at.isoformat(),
                    "updated_at": item.updated_at.isoformat()
                }
                for item in knowledge_items
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get knowledge: {e}")

# Authentication endpoints
@app.post("/auth/token")
async def create_token(username: str, password: str):
    """Create JWT token for authentication"""
    if not config["api"]["authentication"]["enabled"]:
        raise HTTPException(status_code=501, detail="Authentication disabled")
    
    # TODO: Implement proper user authentication
    # For now, accept any credentials in development
    if config["system"]["environment"] == "development":
        payload = {
            "user": username,
            "role": "admin",
            "exp": datetime.utcnow() + timedelta(seconds=config["api"]["authentication"]["token_expiry"])
        }
        
        token = jwt.encode(
            payload,
            config["api"]["authentication"]["jwt_secret"],
            algorithm="HS256"
        )
        
        return {"access_token": token, "token_type": "bearer"}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logging.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "stoll_api:app",
        host=config["api"]["host"],
        port=config["api"]["port"],
        reload=config["system"]["debug"],
        log_level=config["system"]["log_level"].lower()
    )
