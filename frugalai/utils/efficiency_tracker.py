# utils.efficiency_tracker.py

from functools import wraps
from codecarbon import EmissionsTracker
import pandas as pd
from time import strftime, gmtime

class FunctionTracker:
    """Class to track execution time of functions using a decorator."""
    
    _timings = {}
    _emissions = {}
    _energy = {}

    @classmethod
    def track(self, func):
        """Decorator to track function execution time."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            emission_tracker = EmissionsTracker(log_level="error")
            emission_tracker.start()

            result = func(*args, **kwargs)

            execution_emissions = emission_tracker.stop()
            execution_energy_conso = emission_tracker.final_emissions_data.energy_consumed
            execution_time = emission_tracker.final_emissions_data.duration

            self._emissions[func.__name__] = execution_emissions
            self._energy[func.__name__] = execution_energy_conso
            self._timings[func.__name__] = execution_time
            
            formatted_time = strftime("%H:%M:%S", gmtime(execution_time))
            milliseconds = f"{(execution_time % 1):.4f}"[2:]

            print(f"\n⏳ FunctionTimer: {func.__name__}")
            print(f"| {'time':<15} {formatted_time}.{milliseconds}")
            print(f"| {'emissions':<15} {execution_emissions:.6f} CO2eq")
            print(f"| {'energy consumed':<15} {execution_energy_conso:.6f} kWh")
            print()
            return result
        
        return wrapper
    
    @classmethod
    def get_timings(self):
        """Returns the recorded function execution times in seconds."""
        return self._timings

    @classmethod
    def get_emissions(self):
        """Returns the recorded function execution emissions in CO2eq."""
        return self._emissions

    @classmethod
    def get_energy(self):
        """Returns the recorded function execution energy consumed in kWh."""
        return self._energy

    @classmethod
    def get_metrics(self):
        """Returns the recorded metrics."""
        metrics = pd.DataFrame.from_dict({
            'Timings (seconds)': self._timings, 
            'Emissions (CO2eq)': self._emissions, 
            'Energy (kWh)': self._energy
            })
        metrics.loc['Total'] = metrics.sum()
        return metrics


    @classmethod
    def print_timings(self):
        """Prints the stored function timings in a readable format."""
        print("\n📊 Function Execution Times:")
        for func, time_taken in self._timings.items():
            formatted_time = strftime("%H:%M:%S", gmtime(time_taken))
            milliseconds = f"{(time_taken % 1):.4f}"[2:]
            print(f"⚡ {func}: {formatted_time}.{milliseconds}")
        print()

    @classmethod
    def print_emissions(self):
        """Prints the stored function emissions in a readable format."""
        print("\n📊 Function Execution Emissions:")
        for func, emissions in self._emissions.items():
            print(f"⚡ {func}: {emissions:.6f} CO2eq")
        print()

    @classmethod
    def print_energy(self):
        """Prints the stored function energy consumed in a readable format."""
        print("\n📊 Function Execution Energy consumed:")
        for func, energy_consumed in self._energy.items():
            print(f"⚡ {func}: {energy_consumed:.6f} kWh")
        print()


    @classmethod
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


