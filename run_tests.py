#!/usr/bin/env python3
"""
Test Runner for Trident Trading System
Provides comprehensive testing capabilities with different test suites
"""

import unittest
import sys
import os
import time
import argparse
import json
from datetime import datetime

# Add project path
sys.path.append('/workspaces/talos/OCaml')
sys.path.append('/workspaces/talos')

from test_config import get_test_config, setup_test_paths, cleanup_test_paths


class TridentTestRunner:
    """Main test runner for Trident trading system"""
    
    def __init__(self, environment='unit', verbose=True):
        self.environment = environment
        self.verbose = verbose
        self.config = get_test_config(environment)
        self.results = {}
        
    def run_test_suite(self, test_pattern='test_*.py'):
        """Run specific test suite"""
        print(f"🧪 Running {self.environment} tests for Trident Trading System")
        print(f"⚙️  Configuration: {self.config}")
        print("-" * 80)
        
        # Set up test environment
        setup_test_paths()
        
        try:
            # Discover and run tests
            loader = unittest.TestLoader()
            start_dir = '/workspaces/talos'
            suite = loader.discover(start_dir, pattern=test_pattern)
            
            # Run tests with custom result handler
            runner = unittest.TextTestRunner(
                verbosity=2 if self.verbose else 1,
                stream=sys.stdout,
                buffer=True
            )
            
            start_time = time.time()
            result = runner.run(suite)
            end_time = time.time()
            
            # Store results
            self.results = {
                'environment': self.environment,
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
                'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100 if result.testsRun > 0 else 0,
                'execution_time': end_time - start_time,
                'timestamp': datetime.now().isoformat()
            }
            
            # Print summary
            self.print_summary(result)
            
            return result.wasSuccessful()
            
        finally:
            # Clean up test environment
            cleanup_test_paths()
    
    def print_summary(self, result):
        """Print test summary"""
        print("\n" + "="*80)
        print("📊 TEST SUMMARY")
        print("="*80)
        print(f"Environment: {self.environment}")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Success rate: {self.results['success_rate']:.1f}%")
        print(f"Execution time: {self.results['execution_time']:.2f} seconds")
        
        if result.failures:
            print(f"\n❌ FAILURES ({len(result.failures)}):")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback.splitlines()[-1]}")
        
        if result.errors:
            print(f"\n💥 ERRORS ({len(result.errors)}):")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback.splitlines()[-1]}")
        
        if result.wasSuccessful():
            print("\n✅ ALL TESTS PASSED!")
        else:
            print("\n❌ SOME TESTS FAILED!")
    
    def run_specific_test(self, test_class, test_method=None):
        """Run a specific test class or method"""
        if test_method:
            suite = unittest.TestSuite()
            suite.addTest(test_class(test_method))
        else:
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        
        runner = unittest.TextTestRunner(verbosity=2 if self.verbose else 1)
        result = runner.run(suite)
        
        return result.wasSuccessful()
    
    def run_performance_tests(self):
        """Run performance-focused tests"""
        print("🚀 Running performance tests...")
        
        # Import test classes
        from test_trident_system import TestPerformance
        
        # Run performance tests
        suite = unittest.TestLoader().loadTestsFromTestCase(TestPerformance)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
    
    def run_stress_tests(self):
        """Run stress tests"""
        print("💪 Running stress tests...")
        
        # Import test classes
        from test_trident_system import TestStressConditions
        
        # Run stress tests
        suite = unittest.TestLoader().loadTestsFromTestCase(TestStressConditions)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
    
    def run_integration_tests(self):
        """Run integration tests"""
        print("🔗 Running integration tests...")
        
        # Import test classes
        from test_trident_system import TestIntegration
        
        # Run integration tests
        suite = unittest.TestLoader().loadTestsFromTestCase(TestIntegration)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
    
    def save_results(self, filename=None):
        """Save test results to file"""
        if filename is None:
            filename = f"test_results_{self.environment}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"💾 Test results saved to {filename}")
    
    def run_code_coverage(self):
        """Run tests with code coverage"""
        try:
            import coverage
            
            # Start coverage
            cov = coverage.Coverage()
            cov.start()
            
            # Run tests
            success = self.run_test_suite()
            
            # Stop coverage
            cov.stop()
            cov.save()
            
            # Generate report
            print("\n📈 CODE COVERAGE REPORT:")
            cov.report()
            
            return success
            
        except ImportError:
            print("⚠️  Coverage module not available. Run: pip install coverage")
            return self.run_test_suite()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Trident Trading System Test Runner')
    parser.add_argument('--env', choices=['unit', 'integration', 'stress'], 
                       default='unit', help='Test environment')
    parser.add_argument('--suite', choices=['all', 'unit', 'integration', 'performance', 'stress'], 
                       default='all', help='Test suite to run')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--coverage', action='store_true', help='Run with code coverage')
    parser.add_argument('--save', action='store_true', help='Save results to file')
    parser.add_argument('--class', dest='test_class', help='Run specific test class')
    parser.add_argument('--method', dest='test_method', help='Run specific test method')
    
    args = parser.parse_args()
    
    # Create test runner
    runner = TridentTestRunner(environment=args.env, verbose=args.verbose)
    
    success = True
    
    try:
        if args.test_class:
            # Run specific test class
            module_name = 'test_trident_system'
            module = __import__(module_name)
            test_class = getattr(module, args.test_class)
            success = runner.run_specific_test(test_class, args.test_method)
            
        elif args.suite == 'unit':
            success = runner.run_test_suite('test_trident_system.py')
            
        elif args.suite == 'integration':
            success = runner.run_integration_tests()
            
        elif args.suite == 'performance':
            success = runner.run_performance_tests()
            
        elif args.suite == 'stress':
            success = runner.run_stress_tests()
            
        elif args.suite == 'all':
            if args.coverage:
                success = runner.run_code_coverage()
            else:
                success = runner.run_test_suite()
        
        # Save results if requested
        if args.save:
            runner.save_results()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test runner error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
