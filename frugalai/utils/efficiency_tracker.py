# utils.efficiency_tracker.py

import time
from functools import wraps

class FunctionTimer:
    """Class to track execution time of functions using a decorator."""
    
    _timings = {}  # Dictionary to store function execution times
    
    @classmethod
    def track(cls, func):
        """Decorator to track function execution time."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()  # Start timing
            result = func(*args, **kwargs)    # Call the function
            end_time = time.perf_counter()    # End timing
            
            execution_time = end_time - start_time
            cls._timings[func.__name__] = execution_time  # Store in dictionary
            
            print(f"\n⏳ FunctionTimer: {func.__name__} executed in {execution_time // 60:.0f} min {execution_time % 60:.2f} sec\n")
            return result
        
        return wrapper
    
    @classmethod
    def get_timings(cls):
        """Returns the recorded function execution times."""
        return cls._timings

    @classmethod
    def print_timings(cls):
        """Prints the stored function timings in a readable format."""
        print("\n📊 Function Execution Times:")
        for func, time_taken in cls._timings.items():
            print(f"⚡ {func}: {time_taken:.4f} seconds")

