"""
Test Configuration for Trident Trading System
"""

import os
import sys

# Test configuration
TEST_CONFIG = {
    'test_data_size': 1000,
    'mock_symbols': ['SPY', 'MSFT', 'GOOGL'],
    'test_capital': 100000,
    'test_episodes': 10,  # Reduced for faster testing
    'performance_threshold': 30,  # seconds
    'memory_limit_mb': 1000,
    'verbose': True
}

# Test data generation parameters
DATA_CONFIG = {
    'price_start': 100,
    'volatility': 0.1,
    'trend': 0.0001,
    'volume_range': (1000, 10000),
    'features_count': 50
}

# Expected test results
EXPECTED_RESULTS = {
    'min_features': 10,
    'max_training_time': 300,  # 5 minutes max
    'min_portfolio_symbols': 1,
    'max_portfolio_symbols': 10,
    'action_space_size': 7,
    'lookback_period': 60
}

# Test environments
TEST_ENVIRONMENTS = {
    'unit': {
        'data_size': 100,
        'episodes': 5,
        'symbols': ['SPY']
    },
    'integration': {
        'data_size': 1000,
        'episodes': 20,
        'symbols': ['SPY', 'MSFT']
    },
    'stress': {
        'data_size': 10000,
        'episodes': 100,
        'symbols': ['SPY', 'MSFT', 'GOOGL', 'V', 'MCO']
    }
}

# Mock data paths (if needed)
MOCK_DATA_PATHS = {
    'price_data': '/tmp/test_price_data.csv',
    'features': '/tmp/test_features.csv',
    'models': '/tmp/test_models/'
}

# Test utilities
def get_test_config(environment='unit'):
    """Get test configuration for specific environment"""
    base_config = TEST_CONFIG.copy()
    env_config = TEST_ENVIRONMENTS.get(environment, TEST_ENVIRONMENTS['unit'])
    
    base_config.update(env_config)
    return base_config

def setup_test_paths():
    """Set up test paths"""
    for path in MOCK_DATA_PATHS.values():
        if path.endswith('/'):
            os.makedirs(path, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)

def cleanup_test_paths():
    """Clean up test paths"""
    import shutil
    for path in MOCK_DATA_PATHS.values():
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
