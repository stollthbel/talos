"""
Stoll AI Configuration
======================

Configuration settings for the Stoll AI multi-agent system.
"""

import os
from typing import Dict, Any

# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================

STOLL_CONFIG = {
    # System Settings
    "system": {
        "name": "Stoll AI",
        "version": "1.0.0",
        "environment": os.getenv("STOLL_ENV", "development"),
        "debug": os.getenv("STOLL_DEBUG", "true").lower() == "true",
        "log_level": os.getenv("STOLL_LOG_LEVEL", "INFO"),
        "data_dir": os.getenv("STOLL_DATA_DIR", "./data"),
        "max_agents": int(os.getenv("STOLL_MAX_AGENTS", "10")),
        "heartbeat_interval": int(os.getenv("STOLL_HEARTBEAT_INTERVAL", "30")),
        "message_retention_days": int(os.getenv("STOLL_MESSAGE_RETENTION", "30"))
    },
    
    # Memory & Database
    "memory": {
        "database_path": os.getenv("STOLL_DB_PATH", "./data/stoll_memory.db"),
        "backup_interval": int(os.getenv("STOLL_BACKUP_INTERVAL", "3600")),
        "max_memory_entries": int(os.getenv("STOLL_MAX_MEMORY", "100000")),
        "embedding_model": os.getenv("STOLL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        "vector_db_path": os.getenv("STOLL_VECTOR_DB", "./data/stoll_vectors.db")
    },
    
    # Communication
    "communication": {
        "message_queue_size": int(os.getenv("STOLL_QUEUE_SIZE", "1000")),
        "max_message_size": int(os.getenv("STOLL_MAX_MSG_SIZE", "1048576")),  # 1MB
        "timeout_seconds": int(os.getenv("STOLL_TIMEOUT", "30")),
        "retry_attempts": int(os.getenv("STOLL_RETRY_ATTEMPTS", "3")),
        "enable_encryption": os.getenv("STOLL_ENCRYPT", "false").lower() == "true"
    },
    
    # CEO Agent Configuration
    "ceo": {
        "decision_confidence_threshold": float(os.getenv("STOLL_CEO_CONFIDENCE", "0.7")),
        "max_concurrent_decisions": int(os.getenv("STOLL_CEO_MAX_DECISIONS", "5")),
        "risk_tolerance": float(os.getenv("STOLL_CEO_RISK_TOLERANCE", "0.3")),
        "strategic_review_interval": int(os.getenv("STOLL_CEO_REVIEW_INTERVAL", "3600")),
        "alert_thresholds": {
            "critical": float(os.getenv("STOLL_ALERT_CRITICAL", "0.9")),
            "high": float(os.getenv("STOLL_ALERT_HIGH", "0.7")),
            "medium": float(os.getenv("STOLL_ALERT_MEDIUM", "0.5"))
        }
    },
    
    # Trading Agent Configuration
    "trading": {
        "max_portfolio_risk": float(os.getenv("STOLL_MAX_PORTFOLIO_RISK", "0.02")),
        "max_single_trade_risk": float(os.getenv("STOLL_MAX_TRADE_RISK", "0.005")),
        "max_daily_trades": int(os.getenv("STOLL_MAX_DAILY_TRADES", "100")),
        "risk_check_interval": int(os.getenv("STOLL_RISK_CHECK_INTERVAL", "60")),
        "position_timeout": int(os.getenv("STOLL_POSITION_TIMEOUT", "86400")),  # 24 hours
        "supported_symbols": ["SPY", "QQQ", "IWM", "GLD", "TLT", "VXX", "UVXY"],
        "default_timeframes": ["1m", "5m", "15m", "1h", "1d"],
        "signal_engine": {
            "enabled": True,
            "confidence_threshold": float(os.getenv("STOLL_SIGNAL_CONFIDENCE", "0.6")),
            "max_signals_per_minute": int(os.getenv("STOLL_MAX_SIGNALS", "10")),
            "backtest_enabled": os.getenv("STOLL_BACKTEST", "true").lower() == "true"
        }
    },
    
    # SaaS Agent Configuration
    "saas": {
        "talos_capital": {
            "enabled": True,
            "url": os.getenv("TALOS_URL", "http://localhost:5000"),
            "api_key": os.getenv("TALOS_API_KEY", ""),
            "monitoring_interval": int(os.getenv("TALOS_MONITOR_INTERVAL", "300")),
            "auto_scaling": {
                "enabled": os.getenv("TALOS_AUTO_SCALE", "true").lower() == "true",
                "min_instances": int(os.getenv("TALOS_MIN_INSTANCES", "1")),
                "max_instances": int(os.getenv("TALOS_MAX_INSTANCES", "5")),
                "cpu_threshold": float(os.getenv("TALOS_CPU_THRESHOLD", "0.8")),
                "memory_threshold": float(os.getenv("TALOS_MEMORY_THRESHOLD", "0.8"))
            }
        },
        "portfolio_tracker": {
            "enabled": True,
            "url": os.getenv("PORTFOLIO_URL", ""),
            "sync_interval": int(os.getenv("PORTFOLIO_SYNC_INTERVAL", "600"))
        }
    },
    
    # CFO Agent Configuration
    "finance": {
        "accounting": {
            "currency": os.getenv("STOLL_CURRENCY", "USD"),
            "fiscal_year_start": os.getenv("STOLL_FISCAL_START", "01-01"),
            "tax_rate": float(os.getenv("STOLL_TAX_RATE", "0.24")),
            "reporting_frequency": os.getenv("STOLL_REPORTING_FREQ", "daily")
        },
        "portfolio": {
            "rebalance_threshold": float(os.getenv("STOLL_REBALANCE_THRESHOLD", "0.05")),
            "cash_buffer": float(os.getenv("STOLL_CASH_BUFFER", "0.1")),
            "max_concentration": float(os.getenv("STOLL_MAX_CONCENTRATION", "0.2")),
            "risk_free_rate": float(os.getenv("STOLL_RISK_FREE_RATE", "0.05"))
        },
        "reporting": {
            "daily_pnl": True,
            "weekly_summary": True,
            "monthly_report": True,
            "quarterly_review": True,
            "annual_report": True
        }
    },
    
    # COO Agent Configuration (Personal Operations)
    "operations": {
        "personal": {
            "calendar_integration": os.getenv("STOLL_CALENDAR", "google").lower(),
            "email_monitoring": os.getenv("STOLL_EMAIL_MONITOR", "true").lower() == "true",
            "task_management": os.getenv("STOLL_TASK_MANAGER", "todoist").lower(),
            "notification_channels": ["email", "sms", "push"],
            "work_hours": {
                "start": os.getenv("STOLL_WORK_START", "09:00"),
                "end": os.getenv("STOLL_WORK_END", "17:00"),
                "timezone": os.getenv("STOLL_TIMEZONE", "America/New_York")
            }
        },
        "home": {
            "smart_home_integration": os.getenv("STOLL_SMART_HOME", "false").lower() == "true",
            "security_monitoring": os.getenv("STOLL_SECURITY", "true").lower() == "true",
            "energy_optimization": os.getenv("STOLL_ENERGY_OPT", "true").lower() == "true"
        }
    },
    
    # Security Configuration
    "security": {
        "encryption_key": os.getenv("STOLL_ENCRYPTION_KEY", ""),
        "api_rate_limit": int(os.getenv("STOLL_API_RATE_LIMIT", "1000")),
        "max_login_attempts": int(os.getenv("STOLL_MAX_LOGIN_ATTEMPTS", "5")),
        "session_timeout": int(os.getenv("STOLL_SESSION_TIMEOUT", "3600")),
        "audit_logging": os.getenv("STOLL_AUDIT_LOG", "true").lower() == "true",
        "ip_whitelist": os.getenv("STOLL_IP_WHITELIST", "").split(",") if os.getenv("STOLL_IP_WHITELIST") else [],
        "two_factor_auth": os.getenv("STOLL_2FA", "false").lower() == "true"
    },
    
    # External APIs
    "external_apis": {
        "openai": {
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "model": os.getenv("OPENAI_MODEL", "gpt-4"),
            "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "4096")),
            "temperature": float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
        },
        "anthropic": {
            "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
            "model": os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
        },
        "alpha_vantage": {
            "api_key": os.getenv("ALPHA_VANTAGE_API_KEY", ""),
            "base_url": "https://www.alphavantage.co/query"
        },
        "polygon": {
            "api_key": os.getenv("POLYGON_API_KEY", ""),
            "base_url": "https://api.polygon.io"
        },
        "stripe": {
            "secret_key": os.getenv("STRIPE_SECRET_KEY", ""),
            "webhook_secret": os.getenv("STRIPE_WEBHOOK_SECRET", "")
        }
    },
    
    # Monitoring & Alerting
    "monitoring": {
        "enabled": True,
        "metrics_retention_days": int(os.getenv("STOLL_METRICS_RETENTION", "90")),
        "alert_channels": {
            "email": {
                "enabled": os.getenv("STOLL_EMAIL_ALERTS", "true").lower() == "true",
                "smtp_server": os.getenv("STOLL_SMTP_SERVER", ""),
                "smtp_port": int(os.getenv("STOLL_SMTP_PORT", "587")),
                "username": os.getenv("STOLL_EMAIL_USERNAME", ""),
                "password": os.getenv("STOLL_EMAIL_PASSWORD", ""),
                "to_address": os.getenv("STOLL_ALERT_EMAIL", "")
            },
            "slack": {
                "enabled": os.getenv("STOLL_SLACK_ALERTS", "false").lower() == "true",
                "webhook_url": os.getenv("STOLL_SLACK_WEBHOOK", ""),
                "channel": os.getenv("STOLL_SLACK_CHANNEL", "#alerts")
            },
            "discord": {
                "enabled": os.getenv("STOLL_DISCORD_ALERTS", "false").lower() == "true",
                "webhook_url": os.getenv("STOLL_DISCORD_WEBHOOK", "")
            }
        }
    },
    
    # Learning & AI
    "learning": {
        "reinforcement_learning": {
            "enabled": os.getenv("STOLL_RL_ENABLED", "true").lower() == "true",
            "learning_rate": float(os.getenv("STOLL_RL_LEARNING_RATE", "0.001")),
            "discount_factor": float(os.getenv("STOLL_RL_DISCOUNT", "0.95")),
            "epsilon": float(os.getenv("STOLL_RL_EPSILON", "0.1")),
            "batch_size": int(os.getenv("STOLL_RL_BATCH_SIZE", "32")),
            "memory_size": int(os.getenv("STOLL_RL_MEMORY_SIZE", "10000"))
        },
        "model_training": {
            "auto_retrain": os.getenv("STOLL_AUTO_RETRAIN", "true").lower() == "true",
            "retrain_interval": int(os.getenv("STOLL_RETRAIN_INTERVAL", "86400")),  # 24 hours
            "validation_split": float(os.getenv("STOLL_VALIDATION_SPLIT", "0.2")),
            "early_stopping": os.getenv("STOLL_EARLY_STOPPING", "true").lower() == "true"
        }
    }
}

# ============================================================================
# ENVIRONMENT-SPECIFIC CONFIGURATIONS
# ============================================================================

DEVELOPMENT_CONFIG = {
    "system": {
        "debug": True,
        "log_level": "DEBUG"
    },
    "memory": {
        "database_path": "./data/dev_stoll_memory.db"
    },
    "trading": {
        "max_portfolio_risk": 0.01,  # Lower risk in dev
        "max_daily_trades": 10  # Fewer trades in dev
    }
}

PRODUCTION_CONFIG = {
    "system": {
        "debug": False,
        "log_level": "INFO"
    },
    "memory": {
        "database_path": "/opt/stoll/data/stoll_memory.db"
    },
    "security": {
        "audit_logging": True,
        "two_factor_auth": True
    }
}

# ============================================================================
# CONFIGURATION LOADER
# ============================================================================

def get_config(environment: str = None) -> Dict[str, Any]:
    """Get configuration for specified environment"""
    if environment is None:
        environment = os.getenv("STOLL_ENV", "development")
    
    config = STOLL_CONFIG.copy()
    
    if environment == "development":
        # Merge development-specific config
        for key, value in DEVELOPMENT_CONFIG.items():
            if key in config:
                config[key].update(value)
            else:
                config[key] = value
    
    elif environment == "production":
        # Merge production-specific config
        for key, value in PRODUCTION_CONFIG.items():
            if key in config:
                config[key].update(value)
            else:
                config[key] = value
    
    return config

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration"""
    required_keys = [
        "system", "memory", "communication", "ceo", "trading"
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration section: {key}")
    
    # Validate API keys for production
    if config["system"]["environment"] == "production":
        if not config["external_apis"]["openai"]["api_key"]:
            raise ValueError("OpenAI API key is required for production")
    
    return True

# ============================================================================
# AGENT TEMPLATES
# ============================================================================

AGENT_TEMPLATES = {
    "ceo": {
        "name": "Chief Executive Officer",
        "description": "Central command and control agent",
        "responsibilities": [
            "Strategic decision making",
            "Risk management oversight",
            "Resource allocation",
            "System coordination",
            "Performance monitoring"
        ],
        "capabilities": [
            "Multi-agent coordination",
            "Risk assessment",
            "Strategic planning",
            "Decision optimization",
            "Crisis management"
        ]
    },
    
    "trading_cto": {
        "name": "Chief Technology Officer - Trading",
        "description": "Manages all trading operations and strategies",
        "responsibilities": [
            "Trading strategy execution",
            "Risk management",
            "Position monitoring",
            "Market analysis",
            "Signal generation"
        ],
        "capabilities": [
            "Algorithmic trading",
            "Technical analysis",
            "Risk assessment",
            "Portfolio optimization",
            "Real-time monitoring"
        ]
    },
    
    "saas_cto": {
        "name": "Chief Technology Officer - SaaS",
        "description": "Manages SaaS products and infrastructure",
        "responsibilities": [
            "Product development",
            "Infrastructure management",
            "User experience optimization",
            "Performance monitoring",
            "Security management"
        ],
        "capabilities": [
            "DevOps automation",
            "Performance optimization",
            "Security monitoring",
            "User analytics",
            "A/B testing"
        ]
    },
    
    "cfo": {
        "name": "Chief Financial Officer",
        "description": "Manages financial operations and reporting",
        "responsibilities": [
            "Financial planning",
            "Budget management",
            "Risk assessment",
            "Compliance monitoring",
            "Performance reporting"
        ],
        "capabilities": [
            "Financial modeling",
            "Risk analysis",
            "Compliance checking",
            "Report generation",
            "Audit trails"
        ]
    },
    
    "coo": {
        "name": "Chief Operating Officer",
        "description": "Manages personal operations and daily tasks",
        "responsibilities": [
            "Personal task management",
            "Calendar coordination",
            "Home automation",
            "Personal finance",
            "Life optimization"
        ],
        "capabilities": [
            "Task automation",
            "Calendar management",
            "Smart home control",
            "Personal analytics",
            "Lifestyle optimization"
        ]
    }
}

if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    validate_config(config)
    
    print("Configuration loaded successfully!")
    print(f"Environment: {config['system']['environment']}")
    print(f"Debug mode: {config['system']['debug']}")
    print(f"Log level: {config['system']['log_level']}")
