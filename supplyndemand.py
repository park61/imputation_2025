import numpy as np
from conditions import *

# Mask supply function (time-dependent)
def mask_supply(t, type, region, population):
    """
    Time-dependent mask supply function for different regions
    """
    if type=='by_policy':
        return
    elif type=='constant':
        return population*params[region]['initial_infection_rate']
    elif type == 'fast_s-shape':
        max_supply = population  # Maximum supply
        growth_rate = 0.5    # Growth rate
        return max_supply / (1 + np.exp(-growth_rate * (t - 50)))  # Region 1: Fast increase then stabilization
    elif type == 'med_s-shape':
        max_supply = population  # Maximum supply
        growth_rate = 0.03   # Growth rate
        return max_supply / (1 + np.exp(-growth_rate * (t - 100)))  # Region 2: Medium speed increase then stabilization
    elif type == 'linear':
        max_supply = population  # Maximum supply
        growth_rate = 0.03   # Growth rate
        return 100 + 100 * t  # Linear supply
    else:
        base_supply = 1000   # Base supply
        amplitude = 500      # Amplitude
        period = 30          # Period (days)
        oscillation = amplitude * np.sin(2 * np.pi * t / period)
        return base_supply + oscillation  # Region 3: Periodic variation

def mask_demand(pop, I, t):
    """
    Calculate mask demand based on population, infection, and time
    """
    base_demand = 0.9 * pop
    infection_driven_demand = 10 * I
    time_driven_demand = 0.9 * pop * (1 - np.exp(-0.01 * t))
    
    total_demand = (base_demand + infection_driven_demand + time_driven_demand) / mask_durations_fixed
    return np.minimum(total_demand, pop)

def mask_use_rate(pop, I, t, type,region):
    """
    Calculate actual mask usage rate (proportion of population wearing masks)
    """
    supply = mask_supply(t, type, region, pop)
    demand = mask_demand(pop, I, t)
    actual_usage = np.minimum(supply, demand)
    return np.minimum(actual_usage / pop, 1.0)  # Ensure rate doesn't exceed 1.0

