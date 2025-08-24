"""
EVER Test Framework - Simple framework for testing EVER components
"""
import time
import json
import logging
from typing import Dict, List, Any, Callable
import traceback

class EVERTestFramework:
    """
    Testing framework for EVER components
    """
    
    def __init__(self, config=None):
        self.config = {
            'log_level': 'INFO',
            'output_format': 'text',  # 'text' or 'json'
            'save_results': True,
            'output_dir': './test_results/',
            'timeout': 30  # seconds
        }
        
        if config:
            self.config.update(config)
        
        # Configure logging
        log_level = getattr(logging, self.config['log_level'])
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('EVERTest')
        
        # Test results
        self.results = {
            'summary': {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'errors': 0,
                'skipped': 0,
                'total_time': 0
            },
            'tests': []
        }
    
    def run_test_suite(self, test_suite: Dict[str, Callable]) -> Dict:
        """
        Run a test suite
        
        Args:
            test_suite: Dictionary of test_name -> test_function
            
        Returns:
            Test results
        """
        suite_start = time.time()
        self.logger.info(f"Starting test suite with {len(test_suite)} tests")
        
        # Reset results
        self.results = {
            'summary': {
                'total': len(test_suite),
                'passed': 0,
                'failed': 0,
                'errors': 0,
                'skipped': 0,
                'total_time': 0
            },
            'tests': []
        }
        
        # Run each test
        for test_name, test_func in test_suite.items():
            test_result = self.run_test(test_name, test_func)
            self.results['tests'].append(test_result)
            
            # Update summary
            status = test_result['status']
            self.results['summary'][status] += 1
        
        # Calculate total time
        suite_time = time.time() - suite_start
        self.results['summary']['total_time'] = suite_time
        
        # Log summary
        self.logger.info(f"Test suite completed in {suite_time:.2f} seconds")
        self.logger.info(f"Passed: {self.results['summary']['passed']}")
        self.logger.info(f"Failed: {self.results['summary']['failed']}")
        self.logger.info(f"Errors: {self.results['summary']['errors']}")
        self.logger.info(f"Skipped: {self.results['summary']['skipped']}")
        
        # Save results if configured
        if self.config['save_results']:
            self._save_results()
        
        return self.results
    
    def run_test(self, test_name: str, test_func: Callable) -> Dict:
        """
        Run a single test
        
        Args:
            test_name: Name of the test
            test_func: Test function
            
        Returns:
            Test result
        """
        self.logger.info(f"Running test: {test_name}")
        
        result = {
            'name': test_name,
            'status': 'skipped',
            'time': 0,
            'error': None,
            'output': None
        }
        
        try:
            # Time the test
            start_time = time.time()
            
            # Run with timeout
            output = self._run_with_timeout(test_func)
            
            # Calculate time
            end_time = time.time()
            test_time = end_time - start_time
            
            # Update result
            result['time'] = test_time
            result['output'] = output
            
            # Check if test passed
            if output.get('passed', False):
                result['status'] = 'passed'
                self.logger.info(f"Test {test_name} PASSED in {test_time:.2f} seconds")
            else:
                result['status'] = 'failed'
                result['error'] = output.get('error', 'Test returned failed status')
                self.logger.warning(f"Test {test_name} FAILED in {test_time:.2f} seconds")
                if 'error' in output:
                    self.logger.warning(f"Error: {output['error']}")
        
        except Exception as e:
            # Handle errors
            result['status'] = 'errors'
            result['error'] = str(e)
            result['traceback'] = traceback.format_exc()
            self.logger.error(f"Error in test {test_name}: {e}")
        
        return result
    
    def _run_with_timeout(self, test_func: Callable) -> Dict:
        """Run a test function with timeout"""
        # In a more complete implementation, this would use threading or multiprocessing
        # to enforce a timeout. For simplicity, we'll just call the function directly.
        
        try:
            output = test_func()
            
            # Ensure output is a dictionary
            if not isinstance(output, dict):
                output = {'output': output, 'passed': False, 'error': 'Test did not return a dictionary'}
            
            return output
        
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _save_results(self) -> None:
        """Save test results"""
        import os
        import datetime
        
        # Create output directory if it doesn't exist
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"ever_test_results_{timestamp}"
        
        if self.config['output_format'] == 'json':
            filepath = os.path.join(self.config['output_dir'], f"{filename}.json")
            with open(filepath, 'w') as f:
                json.dump(self.results, f, indent=2)
        else:
            filepath = os.path.join(self.config['output_dir'], f"{filename}.txt")
            with open(filepath, 'w') as f:
                f.write(f"EVER Test Results - {timestamp}\n\n")
                f.write(f"Summary:\n")
                f.write(f"  Total tests: {self.results['summary']['total']}\n")
                f.write(f"  Passed: {self.results['summary']['passed']}\n")
                f.write(f"  Failed: {self.results['summary']['failed']}\n")
                f.write(f"  Errors: {self.results['summary']['errors']}\n")
                f.write(f"  Skipped: {self.results['summary']['skipped']}\n")
                f.write(f"  Total time: {self.results['summary']['total_time']:.2f} seconds\n\n")
                
                f.write(f"Test Results:\n")
                for test in self.results['tests']:
                    f.write(f"  {test['name']}:\n")
                    f.write(f"    Status: {test['status']}\n")
                    f.write(f"    Time: {test['time']:.2f} seconds\n")
                    if test['error']:
                        f.write(f"    Error: {test['error']}\n")
        
        self.logger.info(f"Test results saved to {filepath}")