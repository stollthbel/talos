#!/usr/bin/env python3
"""
Simple test to verify the refactored code works
"""

import sys
import os

# Add the path
sys.path.append('/workspaces/talos/OCaml')

def test_imports():
    """Test that all main components can be imported"""
    try:
        from stolltridenthahnbidder_refactored import (
            TradingConstants, HahnPortfolioConstants, 
            TransformerEncoder, TradingEnvironment
        )
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_constants():
    """Test that constants are properly defined"""
    try:
        from stolltridenthahnbidder_refactored import TradingConstants
        
        # Test some key constants
        assert TradingConstants.ACTION_SPACE_SIZE == 7
        assert TradingConstants.LOOKBACK_PERIOD == 60
        assert TradingConstants.INITIAL_CAPITAL == 100000
        assert TradingConstants.TRADE_SIZE_SMALL == 1000
        assert TradingConstants.TRADE_SIZE_MEDIUM == 2500
        assert TradingConstants.TRADE_SIZE_LARGE == 5000
        
        print("✅ Constants test passed")
        return True
    except Exception as e:
        print(f"❌ Constants test failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    try:
        from stolltridenthahnbidder_refactored import TradingConstants, TransformerEncoder
        import torch
        
        # Test transformer encoder
        encoder = TransformerEncoder(input_dim=50)
        x = torch.randn(1, 60, 50)
        
        with torch.no_grad():
            output = encoder(x)
        
        assert output.shape == (1, 60, TradingConstants.TRANSFORMER_D_MODEL)
        print("✅ Basic functionality test passed")
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 Running simple validation tests...")
    
    tests = [
        test_imports,
        test_constants,
        test_basic_functionality
    ]
    
    results = []
    for test in tests:
        print(f"\n🔍 Running {test.__name__}...")
        result = test()
        results.append(result)
    
    print(f"\n📊 Results: {sum(results)}/{len(results)} tests passed")
    
    if all(results):
        print("🎉 All tests passed! Refactored code is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
