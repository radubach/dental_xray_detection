#!/usr/bin/env python3
"""
Test runner for dental_xray_detection project.

This script runs all tests in the tests and tests/datasets directories.
"""

import unittest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_tests():
    """Run all tests in the tests and tests/datasets directories."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Discover tests in the main tests directory
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite.addTests(loader.discover(start_dir, pattern='test_*.py'))
    
    # Discover tests in the datasets subdirectory
    datasets_dir = os.path.join(start_dir, 'datasets')
    if os.path.exists(datasets_dir):
        suite.addTests(loader.discover(datasets_dir, pattern='test_*.py'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1) 