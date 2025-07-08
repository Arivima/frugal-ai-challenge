# utils.efficiency_tracker.py

from functools import wraps
from codecarbon import EmissionsTracker
import pandas as pd
from time import strftime, gmtime

class FunctionTracker:
    """Class to track execution time of functions using a decorator."""
    def __init__(self):
        self._timings = {}
        self._emissions = {}
        self._energy = {}

    def track(self, func):
        """Decorator to track function execution using CodeCarbon tasks."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # track metrics
            tracker = EmissionsTracker(log_level="error")
            tracker.start()

            result = func(*args, **kwargs)

            execution_emissions = tracker.stop()
            execution_energy = tracker.final_emissions_data.energy_consumed
            execution_time = tracker.final_emissions_data.duration
            del tracker

            # record metrics
            self._emissions[func.__name__] = execution_emissions
            self._energy[func.__name__] = execution_energy
            self._timings[func.__name__] = execution_time
            
            # print metrics
            formatted_time = strftime("%H:%M:%S", gmtime(execution_time))
            milliseconds = f"{(execution_time % 1):.4f}"[2:]
            print(f"\n⏳ FunctionTimer: {func.__name__}")
            print(f"| {'time':<15} {formatted_time}.{milliseconds}")
            print(f"| {'emissions':<15} {execution_emissions:.6f} CO2eq")
            print(f"| {'energy consumed':<15} {execution_energy:.6f} kWh")
            print()

            return result
        return wrapper
            
    def get_timings(self):
        """Returns the recorded function execution times in seconds."""
        return self._timings

    def get_emissions(self):
        """Returns the recorded function execution emissions in CO2eq."""
        return self._emissions

    def get_energy(self):
        """Returns the recorded function execution energy consumed in kWh."""
        return self._energy

    def get_metrics(self):
        """Returns the recorded metrics."""
        metrics = pd.DataFrame({
            'Timings (seconds)': self._timings, 
            'Emissions (CO2eq)': self._emissions, 
            'Energy (kWh)': self._energy
        })
        metrics.loc['Total'] = metrics.sum(numeric_only=True)
        return metrics

    def print_timings(self):
        """Prints the stored function timings in a readable format."""
        print("\n📊 Function Execution Times:")
        for func, time_taken in self._timings.items():
            formatted_time = strftime("%H:%M:%S", gmtime(time_taken))
            milliseconds = f"{(time_taken % 1):.4f}"[2:]
            print(f"⚡ {func}: {formatted_time}.{milliseconds}")
        print()

    def print_emissions(self):
        """Prints the stored function emissions in a readable format."""
        print("\n📊 Function Execution Emissions:")
        for func, emissions in self._emissions.items():
            print(f"⚡ {func}: {emissions:.6f} CO2eq")
        print()

    def print_energy(self):
        """Prints the stored function energy consumed in a readable format."""
        print("\n📊 Function Execution Energy consumed:")
        for func, energy_consumed in self._energy.items():
            print(f"⚡ {func}: {energy_consumed:.6f} kWh")
        print()


    def print_metrics(self):
        """Prints the stored metrics in a readable format."""
        print("\n📊 Function Execution Times:")
        metrics = pd.DataFrame.from_dict({
            'Timings (seconds)': self._timings, 
            'Emissions (CO2eq)': self._emissions, 
            'Energy (kWh)': self._energy
            })
        metrics.loc['Total'] = metrics.sum()
        print(metrics)
        print()


if __name__ == '__main__':
    tracker = FunctionTracker()
    import time

    @tracker.track
    def training():
        time.sleep(1)
        return 1
        
    @tracker.track
    def inference():
        time.sleep(1)
        return 1
        
    training()
    tracker.print_metrics()

    inference()
    tracker.print_metrics()
