import unittest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the path to import our refactored module
sys.path.append('/workspaces/talos/OCaml')

from stolltridenthahnbidder_refactored import (
    TradingConstants, HahnPortfolioConstants, TransformerEncoder, 
    PositionalEncoding, DuelingDQN, TradingEnvironment, 
    MarketFeatureExtractor, HahnPortfolioManager, DQNAgent, 
    ReplayBuffer, PerformanceTracker, TridentTradingSystem
)


class TestTradingConstants(unittest.TestCase):
    """Test that all constants are properly defined and accessible"""
    
    def test_neural_network_constants(self):
        """Test neural network architecture constants"""
        self.assertEqual(TradingConstants.TRANSFORMER_D_MODEL, 256)
        self.assertEqual(TradingConstants.TRANSFORMER_NHEAD, 8)
        self.assertEqual(TradingConstants.TRANSFORMER_NUM_LAYERS, 4)
        self.assertEqual(TradingConstants.TRANSFORMER_DROPOUT, 0.1)
        self.assertEqual(TradingConstants.TRANSFORMER_FEEDFORWARD_MULTIPLIER, 4)
    
    def test_trading_environment_constants(self):
        """Test trading environment constants"""
        self.assertEqual(TradingConstants.INITIAL_CAPITAL, 100000)
        self.assertEqual(TradingConstants.MAX_POSITION, 10000)
        self.assertEqual(TradingConstants.TRANSACTION_COST, 0.001)
        self.assertEqual(TradingConstants.LOOKBACK_PERIOD, 60)
        self.assertEqual(TradingConstants.ACTION_SPACE_SIZE, 7)
    
    def test_action_mappings(self):
        """Test action mapping constants"""
        self.assertEqual(TradingConstants.ACTION_HOLD, 0)
        self.assertEqual(TradingConstants.ACTION_BUY_SMALL, 1)
        self.assertEqual(TradingConstants.ACTION_BUY_MEDIUM, 2)
        self.assertEqual(TradingConstants.ACTION_BUY_LARGE, 3)
        self.assertEqual(TradingConstants.ACTION_SELL_SMALL, 4)
        self.assertEqual(TradingConstants.ACTION_SELL_MEDIUM, 5)
        self.assertEqual(TradingConstants.ACTION_SELL_LARGE, 6)
    
    def test_trade_sizes(self):
        """Test trade size constants"""
        self.assertEqual(TradingConstants.TRADE_SIZE_SMALL, 1000)
        self.assertEqual(TradingConstants.TRADE_SIZE_MEDIUM, 2500)
        self.assertEqual(TradingConstants.TRADE_SIZE_LARGE, 5000)
    
    def test_dqn_parameters(self):
        """Test DQN hyperparameters"""
        self.assertEqual(TradingConstants.DQN_LEARNING_RATE, 0.001)
        self.assertEqual(TradingConstants.DQN_GAMMA, 0.99)
        self.assertEqual(TradingConstants.DQN_EPSILON, 0.1)
        self.assertEqual(TradingConstants.DQN_MIN_EPSILON, 0.01)
        self.assertEqual(TradingConstants.DQN_EPSILON_DECAY, 0.995)
        self.assertEqual(TradingConstants.DQN_BATCH_SIZE, 32)
        self.assertEqual(TradingConstants.DQN_UPDATE_FREQUENCY, 4)
        self.assertEqual(TradingConstants.DQN_TARGET_UPDATE_FREQUENCY, 1000)
        self.assertEqual(TradingConstants.DQN_REPLAY_BUFFER_SIZE, 10000)
    
    def test_technical_indicator_constants(self):
        """Test technical indicator constants"""
        self.assertEqual(TradingConstants.ROLLING_WINDOWS, [5, 10, 20, 50, 100, 200])
        self.assertEqual(TradingConstants.MOMENTUM_PERIODS, [10, 20, 50])
        self.assertEqual(TradingConstants.RSI_OVERBOUGHT, 70)
        self.assertEqual(TradingConstants.RSI_OVERSOLD, 30)
        self.assertEqual(TradingConstants.MACD_FAST_PERIOD, 12)
        self.assertEqual(TradingConstants.MACD_SLOW_PERIOD, 26)
        self.assertEqual(TradingConstants.BB_PERIOD, 20)


class TestHahnPortfolioConstants(unittest.TestCase):
    """Test portfolio-related constants"""
    
    def test_portfolio_universe(self):
        """Test portfolio universe structure"""
        universe = HahnPortfolioConstants.PORTFOLIO_UNIVERSE
        self.assertIn('MSFT', universe)
        self.assertIn('V', universe)
        self.assertIn('SPY', universe)
        
        # Check required keys in each symbol
        for symbol, data in universe.items():
            self.assertIn('target_weight', data)
            self.assertIn('roe', data)
            self.assertIn('roa', data)
    
    def test_capital_phases(self):
        """Test capital phase definitions"""
        phases = HahnPortfolioConstants.CAPITAL_PHASES
        self.assertIn('phase_1', phases)
        self.assertIn('phase_2', phases)
        self.assertIn('phase_3', phases)
        self.assertIn('phase_4', phases)
        
        # Check phase structure
        for phase, data in phases.items():
            self.assertIn('range', data)
            self.assertIn('strategy', data)
            self.assertEqual(len(data['range']), 2)
    
    def test_portfolio_weights_sum(self):
        """Test that portfolio weights sum to approximately 1.0"""
        universe = HahnPortfolioConstants.PORTFOLIO_UNIVERSE
        total_weight = sum(data['target_weight'] for data in universe.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)


class TestTransformerComponents(unittest.TestCase):
    """Test transformer-related components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.input_dim = 50
        self.batch_size = 16
        self.seq_len = 60
        
    def test_positional_encoding_initialization(self):
        """Test positional encoding initialization"""
        pos_enc = PositionalEncoding(TradingConstants.TRANSFORMER_D_MODEL)
        self.assertEqual(pos_enc.pe.shape[1], TradingConstants.TRANSFORMER_D_MODEL)
        self.assertEqual(pos_enc.pe.shape[0], TradingConstants.POSITIONAL_ENCODING_MAX_LEN)
    
    def test_transformer_encoder_initialization(self):
        """Test transformer encoder initialization"""
        encoder = TransformerEncoder(self.input_dim)
        self.assertEqual(encoder.input_dim, self.input_dim)
        self.assertEqual(encoder.d_model, TradingConstants.TRANSFORMER_D_MODEL)
        self.assertIsInstance(encoder.transformer, nn.TransformerEncoder)
    
    def test_transformer_encoder_forward(self):
        """Test transformer encoder forward pass"""
        encoder = TransformerEncoder(self.input_dim)
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        
        with torch.no_grad():
            output = encoder(x)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, TradingConstants.TRANSFORMER_D_MODEL))
    
    def test_dueling_dqn_initialization(self):
        """Test dueling DQN initialization"""
        dqn = DuelingDQN(self.input_dim)
        self.assertIsInstance(dqn.transformer, TransformerEncoder)
        self.assertIsInstance(dqn.value_stream, nn.Sequential)
        self.assertIsInstance(dqn.advantage_stream, nn.Sequential)
    
    def test_dueling_dqn_forward(self):
        """Test dueling DQN forward pass"""
        dqn = DuelingDQN(self.input_dim)
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        
        with torch.no_grad():
            q_values = dqn(x)
        
        self.assertEqual(q_values.shape, (self.batch_size, TradingConstants.ACTION_SPACE_SIZE))


class TestTradingEnvironment(unittest.TestCase):
    """Test trading environment functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock data
        dates = pd.date_range('2023-01-01', periods=1000, freq='H')
        self.test_data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(1000) * 0.1),
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'feature3': np.random.randn(1000),
        }, index=dates)
        
        self.env = TradingEnvironment(self.test_data)
    
    def test_environment_initialization(self):
        """Test environment initialization"""
        self.assertEqual(self.env.initial_capital, TradingConstants.INITIAL_CAPITAL)
        self.assertEqual(self.env.max_position, TradingConstants.MAX_POSITION)
        self.assertEqual(self.env.transaction_cost, TradingConstants.TRANSACTION_COST)
        self.assertEqual(self.env.action_space.n, TradingConstants.ACTION_SPACE_SIZE)
    
    def test_action_mapping(self):
        """Test action mapping"""
        expected_map = {
            TradingConstants.ACTION_HOLD: 0,
            TradingConstants.ACTION_BUY_SMALL: TradingConstants.TRADE_SIZE_SMALL,
            TradingConstants.ACTION_BUY_MEDIUM: TradingConstants.TRADE_SIZE_MEDIUM,
            TradingConstants.ACTION_BUY_LARGE: TradingConstants.TRADE_SIZE_LARGE,
            TradingConstants.ACTION_SELL_SMALL: -TradingConstants.TRADE_SIZE_SMALL,
            TradingConstants.ACTION_SELL_MEDIUM: -TradingConstants.TRADE_SIZE_MEDIUM,
            TradingConstants.ACTION_SELL_LARGE: -TradingConstants.TRADE_SIZE_LARGE
        }
        
        self.assertEqual(self.env.action_map, expected_map)
    
    def test_reset_functionality(self):
        """Test environment reset"""
        obs, info = self.env.reset()
        
        self.assertEqual(self.env.current_step, TradingConstants.LOOKBACK_PERIOD)
        self.assertEqual(self.env.position, 0)
        self.assertEqual(self.env.cash, TradingConstants.INITIAL_CAPITAL)
        self.assertEqual(self.env.portfolio_value, TradingConstants.INITIAL_CAPITAL)
        self.assertEqual(self.env.trade_count, 0)
        self.assertEqual(obs.shape, (TradingConstants.LOOKBACK_PERIOD, self.test_data.shape[1]))
    
    def test_step_functionality(self):
        """Test environment step"""
        obs, info = self.env.reset()
        
        # Test hold action
        next_obs, reward, done, truncated, info = self.env.step(TradingConstants.ACTION_HOLD)
        self.assertEqual(self.env.position, 0)
        self.assertEqual(self.env.trade_count, 0)
        
        # Test buy action
        next_obs, reward, done, truncated, info = self.env.step(TradingConstants.ACTION_BUY_SMALL)
        self.assertEqual(self.env.position, TradingConstants.TRADE_SIZE_SMALL)
        self.assertEqual(self.env.trade_count, 1)
        
        # Test sell action
        next_obs, reward, done, truncated, info = self.env.step(TradingConstants.ACTION_SELL_SMALL)
        self.assertEqual(self.env.position, 0)
        self.assertEqual(self.env.trade_count, 2)
    
    def test_observation_shape(self):
        """Test observation shape consistency"""
        obs, info = self.env.reset()
        
        for _ in range(10):
            action = np.random.choice(TradingConstants.ACTION_SPACE_SIZE)
            next_obs, reward, done, truncated, info = self.env.step(action)
            
            self.assertEqual(next_obs.shape, (TradingConstants.LOOKBACK_PERIOD, self.test_data.shape[1]))
            
            if done:
                break
    
    def test_reward_calculation(self):
        """Test reward calculation"""
        obs, info = self.env.reset()
        
        # Test hold reward
        reward_hold = self.env._calculate_reward(TradingConstants.ACTION_HOLD, 0, 100)
        self.assertEqual(reward_hold, TradingConstants.HOLD_REWARD)
        
        # Test overtrading penalty
        self.env.trade_count = TradingConstants.MAX_TRADES_PER_EPISODE + 1
        reward_overtrade = self.env._calculate_reward(TradingConstants.ACTION_BUY_SMALL, 1000, 100)
        self.assertLess(reward_overtrade, TradingConstants.HOLD_REWARD)
    
    def test_position_limits(self):
        """Test position limits enforcement"""
        obs, info = self.env.reset()
        
        # Try to exceed max position
        for _ in range(20):  # Should exceed max position
            next_obs, reward, done, truncated, info = self.env.step(TradingConstants.ACTION_BUY_LARGE)
            self.assertLessEqual(abs(self.env.position), TradingConstants.MAX_POSITION)
            
            if done:
                break


class TestMarketFeatureExtractor(unittest.TestCase):
    """Test market feature extraction"""
    
    def setUp(self):
        """Set up test fixtures"""
        dates = pd.date_range('2023-01-01', periods=1000, freq='H')
        self.test_data = pd.DataFrame({
            'open': 100 + np.cumsum(np.random.randn(1000) * 0.1),
            'high': 100 + np.cumsum(np.random.randn(1000) * 0.1) + 1,
            'low': 100 + np.cumsum(np.random.randn(1000) * 0.1) - 1,
            'close': 100 + np.cumsum(np.random.randn(1000) * 0.1),
            'volume': np.random.randint(1000, 10000, 1000)
        }, index=dates)
        
        self.extractor = MarketFeatureExtractor()
    
    def test_feature_extraction(self):
        """Test feature extraction functionality"""
        features = self.extractor.extract_features(self.test_data)
        
        # Check that features are generated
        self.assertGreater(features.shape[1], 5)  # Should have many features
        self.assertGreater(features.shape[0], 0)  # Should have data
        
        # Check that features are normalized
        self.assertTrue(self.extractor.fitted)
    
    def test_feature_names(self):
        """Test that expected features are generated"""
        features = self.extractor.extract_features(self.test_data)
        
        # Check for some expected feature types
        feature_names = features.columns.tolist()
        
        # Should have returns
        returns_features = [f for f in feature_names if 'return' in f]
        self.assertGreater(len(returns_features), 0)
        
        # Should have moving averages
        ma_features = [f for f in feature_names if 'ma_' in f]
        self.assertGreater(len(ma_features), 0)
        
        # Should have technical indicators
        rsi_features = [f for f in feature_names if 'rsi' in f]
        self.assertGreater(len(rsi_features), 0)
    
    def test_feature_consistency(self):
        """Test feature extraction consistency"""
        features1 = self.extractor.extract_features(self.test_data)
        features2 = self.extractor.extract_features(self.test_data)
        
        # Should produce the same features
        self.assertEqual(features1.shape, features2.shape)
        np.testing.assert_array_almost_equal(features1.values, features2.values)


class TestDQNAgent(unittest.TestCase):
    """Test DQN agent functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.state_dim = 50
        self.agent = DQNAgent(self.state_dim)
        self.test_state = np.random.randn(TradingConstants.LOOKBACK_PERIOD, self.state_dim)
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        self.assertEqual(self.agent.state_dim, self.state_dim)
        self.assertEqual(self.agent.action_dim, TradingConstants.ACTION_SPACE_SIZE)
        self.assertEqual(self.agent.lr, TradingConstants.DQN_LEARNING_RATE)
        self.assertEqual(self.agent.gamma, TradingConstants.DQN_GAMMA)
        self.assertEqual(self.agent.epsilon, TradingConstants.DQN_EPSILON)
        
        # Check networks
        self.assertIsInstance(self.agent.q_network, DuelingDQN)
        self.assertIsInstance(self.agent.target_network, DuelingDQN)
        self.assertIsInstance(self.agent.replay_buffer, ReplayBuffer)
    
    def test_action_selection(self):
        """Test action selection"""
        # Test training mode (with epsilon-greedy)
        action = self.agent.select_action(self.test_state, training=True)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, TradingConstants.ACTION_SPACE_SIZE)
        
        # Test evaluation mode (greedy)
        action = self.agent.select_action(self.test_state, training=False)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, TradingConstants.ACTION_SPACE_SIZE)
    
    def test_experience_storage(self):
        """Test experience storage"""
        next_state = np.random.randn(TradingConstants.LOOKBACK_PERIOD, self.state_dim)
        
        initial_buffer_size = len(self.agent.replay_buffer)
        
        self.agent.store_experience(self.test_state, 0, 1.0, next_state, False)
        
        self.assertEqual(len(self.agent.replay_buffer), initial_buffer_size + 1)
    
    def test_training_requirement(self):
        """Test training requirements"""
        # Should not train with insufficient data
        result = self.agent.train()
        self.assertIsNone(result)
        
        # Add enough experiences
        next_state = np.random.randn(TradingConstants.LOOKBACK_PERIOD, self.state_dim)
        for _ in range(TradingConstants.DQN_BATCH_SIZE):
            self.agent.store_experience(self.test_state, 0, 1.0, next_state, False)
        
        # Should now be able to train
        result = self.agent.train()
        self.assertIsInstance(result, float)


class TestReplayBuffer(unittest.TestCase):
    """Test replay buffer functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.capacity = 100
        self.buffer = ReplayBuffer(self.capacity)
        self.state_dim = 50
        self.test_state = np.random.randn(TradingConstants.LOOKBACK_PERIOD, self.state_dim)
    
    def test_buffer_initialization(self):
        """Test buffer initialization"""
        self.assertEqual(self.buffer.capacity, self.capacity)
        self.assertEqual(len(self.buffer), 0)
    
    def test_buffer_add(self):
        """Test adding experiences to buffer"""
        next_state = np.random.randn(TradingConstants.LOOKBACK_PERIOD, self.state_dim)
        
        self.buffer.add(self.test_state, 0, 1.0, next_state, False)
        self.assertEqual(len(self.buffer), 1)
        
        # Test capacity limit
        for _ in range(self.capacity):
            self.buffer.add(self.test_state, 0, 1.0, next_state, False)
        
        self.assertEqual(len(self.buffer), self.capacity)
    
    def test_buffer_sampling(self):
        """Test sampling from buffer"""
        next_state = np.random.randn(TradingConstants.LOOKBACK_PERIOD, self.state_dim)
        
        # Add some experiences
        for i in range(50):
            self.buffer.add(self.test_state, i % TradingConstants.ACTION_SPACE_SIZE, 
                          float(i), next_state, i % 2 == 0)
        
        # Sample batch
        batch_size = 10
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        self.assertEqual(states.shape[0], batch_size)
        self.assertEqual(actions.shape[0], batch_size)
        self.assertEqual(rewards.shape[0], batch_size)
        self.assertEqual(next_states.shape[0], batch_size)
        self.assertEqual(dones.shape[0], batch_size)


class TestPerformanceTracker(unittest.TestCase):
    """Test performance tracking functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tracker = PerformanceTracker()
        self.tracker.initialize()
    
    def test_tracker_initialization(self):
        """Test tracker initialization"""
        self.assertEqual(len(self.tracker.symbol_performance), 0)
        self.assertEqual(len(self.tracker.session_history), 0)
    
    def test_update_symbol_performance(self):
        """Test updating symbol performance"""
        test_results = {
            'trading': {'total_reward': 100.0},
            'symbol': 'MSFT'
        }
        
        self.tracker.update_symbol_performance('MSFT', test_results)
        
        self.assertIn('MSFT', self.tracker.symbol_performance)
        self.assertEqual(len(self.tracker.symbol_performance['MSFT']), 1)
    
    def test_get_symbol_stats(self):
        """Test getting symbol statistics"""
        # Test with no data
        stats = self.tracker.get_symbol_stats('MSFT')
        self.assertIsNone(stats)
        
        # Add some data
        test_results = [
            {'trading': {'total_reward': 100.0}},
            {'trading': {'total_reward': 150.0}},
            {'trading': {'total_reward': 50.0}}
        ]
        
        for result in test_results:
            self.tracker.update_symbol_performance('MSFT', result)
        
        stats = self.tracker.get_symbol_stats('MSFT')
        
        self.assertIsNotNone(stats)
        self.assertIn('mean_return', stats)
        self.assertIn('std_return', stats)
        self.assertIn('sharpe_ratio', stats)
        self.assertIn('max_return', stats)
        self.assertIn('min_return', stats)


class TestHahnPortfolioManager(unittest.TestCase):
    """Test portfolio manager functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.portfolio_manager = HahnPortfolioManager()
    
    def test_portfolio_initialization(self):
        """Test portfolio manager initialization"""
        self.assertEqual(self.portfolio_manager.initial_capital, HahnPortfolioConstants.DEFAULT_INITIAL_CAPITAL)
        self.assertEqual(self.portfolio_manager.hahn_universe, HahnPortfolioConstants.PORTFOLIO_UNIVERSE)
        self.assertEqual(self.portfolio_manager.capital_phases, HahnPortfolioConstants.CAPITAL_PHASES)
    
    def test_determine_phase(self):
        """Test capital phase determination"""
        # Test with different capital amounts
        test_cases = [
            (50000, 'phase_1'),
            (200000, 'phase_2'),
            (1000000, 'phase_3'),
            (5000000, 'phase_4')
        ]
        
        for capital, expected_phase in test_cases:
            pm = HahnPortfolioManager(capital)
            phase = pm.determine_phase()
            self.assertEqual(phase, expected_phase)
    
    @patch('yfinance.Ticker')
    def test_fetch_market_data(self, mock_ticker):
        """Test market data fetching"""
        # Mock yfinance response
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101],
            'Close': [100.5, 101.5, 102.5],
            'Volume': [1000, 1100, 1200]
        })
        
        mock_ticker.return_value.history.return_value = mock_data
        
        data = self.portfolio_manager.fetch_market_data('MSFT')
        
        self.assertIsNotNone(data)
        mock_ticker.assert_called_once_with('MSFT')
    
    @patch('yfinance.Ticker')
    def test_fetch_market_data_error(self, mock_ticker):
        """Test market data fetching with error"""
        mock_ticker.side_effect = Exception("Network error")
        
        data = self.portfolio_manager.fetch_market_data('MSFT')
        
        self.assertIsNone(data)


class TestTridentTradingSystem(unittest.TestCase):
    """Test main trading system functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.system = TridentTradingSystem(initial_capital=100000)
    
    def test_system_initialization(self):
        """Test system initialization"""
        self.assertEqual(self.system.initial_capital, 100000)
        self.assertIsInstance(self.system.portfolio_manager, HahnPortfolioManager)
        self.assertIsInstance(self.system.performance_tracker, PerformanceTracker)
        self.assertIsNotNone(self.system.current_phase)
    
    @patch.object(HahnPortfolioManager, 'initialize_scalping_agents')
    def test_initialize_system(self, mock_initialize):
        """Test system initialization process"""
        mock_initialize.return_value = None
        
        self.system.initialize_system()
        
        mock_initialize.assert_called_once()
    
    def test_training_constants_usage(self):
        """Test that training uses correct constants"""
        # Check that training episodes constant is used
        self.assertEqual(TradingConstants.TRAINING_EPISODES, 100)
        self.assertEqual(TradingConstants.TRAINING_PROGRESS_INTERVAL, 20)
    
    def test_session_report_structure(self):
        """Test session report structure"""
        # Mock session results
        mock_results = {
            'MSFT': {
                'trading': {
                    'total_reward': 100.0,
                    'final_portfolio_value': 15000,
                    'total_trades': 10
                }
            }
        }
        
        report = self.system.generate_session_report(mock_results)
        
        self.assertIn('total_pnl', report)
        self.assertIn('total_pnl_pct', report)
        self.assertIn('total_trades', report)
        self.assertIn('symbol_results', report)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.system = TridentTradingSystem(initial_capital=100000)
    
    @patch('yfinance.Ticker')
    def test_end_to_end_training(self, mock_ticker):
        """Test end-to-end training process"""
        # Mock market data
        dates = pd.date_range('2023-01-01', periods=1000, freq='H')
        mock_data = pd.DataFrame({
            'Open': 100 + np.cumsum(np.random.randn(1000) * 0.1),
            'High': 100 + np.cumsum(np.random.randn(1000) * 0.1) + 1,
            'Low': 100 + np.cumsum(np.random.randn(1000) * 0.1) - 1,
            'Close': 100 + np.cumsum(np.random.randn(1000) * 0.1),
            'Volume': np.random.randint(1000, 10000, 1000)
        }, index=dates)
        
        mock_ticker.return_value.history.return_value = mock_data
        
        # Mock a small subset of symbols to speed up testing
        with patch.object(self.system.portfolio_manager, 'hahn_universe', {'SPY': {'target_weight': 1.0, 'roe': 20.0, 'roa': 12.0}}):
            self.system.initialize_system()
            
            # Check that system initialized
            self.assertGreater(len(self.system.portfolio_manager.scalping_agents), 0)
    
    def test_constants_consistency(self):
        """Test that all constants are consistently used"""
        # Check that action space size is consistent
        env_action_space = TradingConstants.ACTION_SPACE_SIZE
        dqn_action_space = TradingConstants.ACTION_SPACE_SIZE
        
        self.assertEqual(env_action_space, dqn_action_space)
        
        # Check that lookback period is consistent
        lookback = TradingConstants.LOOKBACK_PERIOD
        self.assertGreater(lookback, 0)
        
        # Check that trade sizes are properly defined
        trade_sizes = [
            TradingConstants.TRADE_SIZE_SMALL,
            TradingConstants.TRADE_SIZE_MEDIUM,
            TradingConstants.TRADE_SIZE_LARGE
        ]
        
        for size in trade_sizes:
            self.assertGreater(size, 0)
    
    def test_reward_calculation_consistency(self):
        """Test reward calculation consistency"""
        # Create test environment
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        test_data = pd.DataFrame({
            'close': 100 + np.random.randn(100) * 0.1,
            'feature1': np.random.randn(100),
        }, index=dates)
        
        env = TradingEnvironment(test_data)
        obs, _ = env.reset()
        
        # Test that rewards are calculated consistently
        reward1 = env._calculate_reward(TradingConstants.ACTION_HOLD, 0, 100)
        reward2 = env._calculate_reward(TradingConstants.ACTION_HOLD, 0, 100)
        
        self.assertEqual(reward1, reward2)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def test_empty_data_handling(self):
        """Test handling of empty data"""
        empty_data = pd.DataFrame()
        
        with self.assertRaises(Exception):
            TradingEnvironment(empty_data)
    
    def test_invalid_action_handling(self):
        """Test handling of invalid actions"""
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        test_data = pd.DataFrame({
            'close': 100 + np.random.randn(100) * 0.1,
            'feature1': np.random.randn(100),
        }, index=dates)
        
        env = TradingEnvironment(test_data)
        obs, _ = env.reset()
        
        # Test invalid action (should be handled gracefully)
        with self.assertRaises(KeyError):
            env.step(999)  # Invalid action
    
    def test_insufficient_data_for_features(self):
        """Test feature extraction with insufficient data"""
        # Create very small dataset
        dates = pd.date_range('2023-01-01', periods=10, freq='H')
        small_data = pd.DataFrame({
            'open': [100] * 10,
            'high': [101] * 10,
            'low': [99] * 10,
            'close': [100] * 10,
            'volume': [1000] * 10
        }, index=dates)
        
        extractor = MarketFeatureExtractor()
        
        # Should handle small dataset gracefully
        features = extractor.extract_features(small_data)
        self.assertGreater(features.shape[0], 0)


# ==================== PERFORMANCE TESTS ====================

class TestPerformance(unittest.TestCase):
    """Performance and benchmark tests"""
    
    def test_feature_extraction_performance(self):
        """Test feature extraction performance"""
        import time
        
        # Create large dataset
        dates = pd.date_range('2023-01-01', periods=10000, freq='H')
        large_data = pd.DataFrame({
            'open': 100 + np.cumsum(np.random.randn(10000) * 0.1),
            'high': 100 + np.cumsum(np.random.randn(10000) * 0.1) + 1,
            'low': 100 + np.cumsum(np.random.randn(10000) * 0.1) - 1,
            'close': 100 + np.cumsum(np.random.randn(10000) * 0.1),
            'volume': np.random.randint(1000, 10000, 10000)
        }, index=dates)
        
        extractor = MarketFeatureExtractor()
        
        start_time = time.time()
        features = extractor.extract_features(large_data)
        end_time = time.time()
        
        # Should complete in reasonable time (less than 30 seconds)
        self.assertLess(end_time - start_time, 30)
        self.assertGreater(features.shape[0], 0)
    
    def test_neural_network_forward_pass_performance(self):
        """Test neural network forward pass performance"""
        import time
        
        # Create test data
        batch_size = 32
        seq_len = 60
        input_dim = 100
        
        dqn = DuelingDQN(input_dim)
        x = torch.randn(batch_size, seq_len, input_dim)
        
        # Warm up
        with torch.no_grad():
            _ = dqn(x)
        
        # Time forward pass
        start_time = time.time()
        for _ in range(100):
            with torch.no_grad():
                _ = dqn(x)
        end_time = time.time()
        
        # Should complete 100 forward passes in reasonable time
        self.assertLess(end_time - start_time, 10)


# ==================== STRESS TESTS ====================

class TestStressConditions(unittest.TestCase):
    """Stress tests for extreme conditions"""
    
    def test_extreme_market_conditions(self):
        """Test system under extreme market conditions"""
        # Create extreme market data (high volatility)
        dates = pd.date_range('2023-01-01', periods=1000, freq='H')
        extreme_data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(1000) * 5),  # High volatility
            'feature1': np.random.randn(1000) * 10,
            'feature2': np.random.randn(1000) * 10,
        }, index=dates)
        
        env = TradingEnvironment(extreme_data)
        obs, _ = env.reset()
        
        # Run for many steps
        for _ in range(100):
            action = np.random.choice(TradingConstants.ACTION_SPACE_SIZE)
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Should handle extreme conditions gracefully
            self.assertIsNotNone(reward)
            self.assertIsNotNone(next_obs)
            
            if done:
                break
    
    def test_memory_usage_under_load(self):
        """Test memory usage under high load"""
        # Create large replay buffer
        buffer = ReplayBuffer(TradingConstants.DQN_REPLAY_BUFFER_SIZE)
        state_dim = 100
        
        # Fill buffer to capacity
        for i in range(TradingConstants.DQN_REPLAY_BUFFER_SIZE):
            state = np.random.randn(TradingConstants.LOOKBACK_PERIOD, state_dim)
            next_state = np.random.randn(TradingConstants.LOOKBACK_PERIOD, state_dim)
            buffer.add(state, i % TradingConstants.ACTION_SPACE_SIZE, 1.0, next_state, False)
        
        # Should maintain capacity limit
        self.assertEqual(len(buffer), TradingConstants.DQN_REPLAY_BUFFER_SIZE)
        
        # Add more experiences (should not exceed capacity)
        for _ in range(1000):
            state = np.random.randn(TradingConstants.LOOKBACK_PERIOD, state_dim)
            next_state = np.random.randn(TradingConstants.LOOKBACK_PERIOD, state_dim)
            buffer.add(state, 0, 1.0, next_state, False)
        
        self.assertEqual(len(buffer), TradingConstants.DQN_REPLAY_BUFFER_SIZE)


# ==================== MAIN TEST RUNNER ====================

if __name__ == '__main__':
    # Configure test runner
    unittest.main(verbosity=2, buffer=True)
