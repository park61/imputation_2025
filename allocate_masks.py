import numpy as np
import pandas as pd

from gurobipy import Model, GRB
import gurobipy as gp
import os
import sys
import conditions as conditions

def solve_problem_gurobi(epsilon, supply, total_population, num_breaks, sub_population, slope, max_utility):
    ''' Solve the optimization problem using Gurobi.'''
    
    m = gp.Env(empty=True)
    m.setParam('WLSACCESSID', '9169a63f-4098-4e21-ac54-e82dc75bd51a')#lim
    m.setParam('WLSSECRET', 'e47486a1-410b-4e36-8242-bb1a71df8cde')#lim
    m.setParam('LICENSEID', 2696148)#lim
    m.setParam("OutputFlag", 0) #add by CHP
    m.start()

    #print("epsilon: "+str(epsilon))
    #print("supply: " + str(supply))

    flag = 1  # if optimal solution is found: 0, otherwise 1.
    numPharmacies = total_population.shape[0]  # number of pharmacies
    N = [i for i in range(numPharmacies)]  # define set
    M = [i for i in range(int(num_breaks[0]))]  # define set for break points
    A = [(i, t) for i in range(numPharmacies) for t in range(int(num_breaks[0]))]  # define set for x_i^t
    B = [(j, k) for j in range(numPharmacies) for k in range(numPharmacies) if k > j]

    #mdl = Model("lp")
    mdl = gp.Model(env=m)

    # define variables
    x = mdl.addVars(A, name='x', lb=0.0, vtype=GRB.INTEGER)
    d_plus = mdl.addVars(B, name='d_plus', lb=0.0, vtype=GRB.INTEGER)
    d_minus = mdl.addVars(B, name='d_minus', lb=0.0, vtype=GRB.INTEGER)
    Y = mdl.addVar(name='Y', lb=0.0, vtype=GRB.CONTINUOUS)

    # define objective function
    mdl.setObjective(sum(slope[i, t] * x[i, t] for i, t in A), GRB.MAXIMIZE) #2024.12.26 I added additional "+Y" for generating non-dominated solutions #+Y for only fair. remove for utility

    # define constraints
    mdl.addConstr(sum(x[i, t] for i, t in A) <= supply, "constraint1")
    for i, t in A:
        mdl.addConstr(x[i, t] <= sub_population[i, t])
    for i in N:
        mdl.addConstr(sum(x[i, t] for t in M) <= total_population[i], "constraint2_%d" % i)
    for j, k in B:
        mdl.addConstr(d_plus[j, k] - d_minus[j, k] == max_utility[j] * sum(slope[k, t] * x[k, t] for t in M)
                      - max_utility[k] * sum(slope[j, t] * x[j, t] for t in M), "constraint3_%d_%d" % (j, k))
    mdl.addConstr(sum(d_plus[j, k] + d_minus[j, k] for j, k in B)  <= epsilon
                  * sum(max_utility[i] for i in N) * sum(slope[i, t] * x[i, t] for i, t in A))

    mdl.setParam("OutputFlag", 0)  # Suppress Gurobi output

    mdl.optimize()

    #x_sol = {i: {t: x[i, t].X for t in range(int(num_breaks[0]))} for i in range(numPharmacies)}
    x_sol = {(i, t): x[i, t].X for i, t in A}
    x_sol_list = list(x_sol.values())
    d_plus = {(j, k): d_plus[j, k].X for j, k in B}
    d_plus_list = list(d_plus.values())
    d_minus = {(j, k): d_minus[j, k].X for j, k in B}
    d_minus_list = list(d_minus.values())
    flag = mdl.Status
    optObj = mdl.ObjVal

    return x_sol, d_plus, d_minus, optObj, flag

def restore_values(numPharmacies,x,d_plus,d_minus,obj_val,max_utility,slope):
    ''''''
    num_value = 0
    for j in range(numPharmacies):
        for k in range(numPharmacies):
            if k>j:
                num_value = num_value + d_plus[(j,k)] + d_minus[(j,k)]
    #print(num_value)
    denom_value_1 = 0
    for i in range(numPharmacies):
        denom_value_1 = denom_value_1 + max_utility[i]
    #print(denom_value_1)
    denom_value_2 = 0
    for i in range(numPharmacies):
        for t in range(int(num_breaks[0])):
            denom_value_2 = denom_value_2 + slope[i,t]*x[(i,t)]
    #print(denom_value_2)
    gini = num_value/(denom_value_1*denom_value_2)
    # print(gini)
    return gini, obj_val

def get_utility_value(sub_population,slope,allocation):
  #print(allocation)
  achieved_utility = np.zeros(len(allocation))
  #print(achieved_utility)
  for i in range(len(allocation)):
    if allocation[i] <= sub_population[i,0]:
      achieved_utility[i] = allocation[i] * slope[i,0]
    else:
      achieved_utility[i] = sub_population[i,0]*slope[i,0]+(allocation[i]-sub_population[i,0]) * slope[i,1]

    total_achieved_utility = np.sum(achieved_utility)
    #print(total_achieved_utility)
  return achieved_utility, total_achieved_utility

def get_gini_value(sub_population,slope,allocation,max_utility):
  achieved_gini = 0
  achieved_utility, total_achieved_utility = get_utility_value(sub_population,slope,allocation)

  n = len(allocation)
  temp_numerator_value = 0

  for i in range(n):
    for j in range(n):
      if j>i:
        #print(i,j)
        temp_numerator_value = temp_numerator_value+ abs(max_utility[j]*achieved_utility[i] - max_utility[i]*achieved_utility[j])
  #print (temp_numerator_value)

  temp_denominator_value = 0
  temp_denominator_value = sum(max_utility[i] for i in range(n))*sum(achieved_utility[i] for i in range(n))

  gini = temp_numerator_value/ temp_denominator_value if temp_denominator_value != 0 else np.nan
  
  return gini

def process_allocations(allocations, allocation_type, sub_population, slope, max_utility, supply_rate, data, results):
    """
    Process a list of allocations and append results to the results list.

    Args:
        allocations (list): List of allocation pairs.
        allocation_type (str): Type of allocation (e.g., 'Equal', 'Proportion').
        sub_population (list): Sub-population data.
        slope (float): Slope value for utility calculation.
        max_utility (float): Maximum utility value.
        results (list): List to store the results.
    """
    region_population = data["region_population"]

    for i, allocation in enumerate(allocations):
        # Check if allocation exceeds region_population
        if any(allocation[j] > region_population[j] for j in range(len(region_population))):
            results.append({
                # 배분 결과가 하나일 경우 숫자 없이 시나리오 이름 생성
                "Scenario": allocation_type if len(allocations) == 1 else f"{allocation_type} {i + 1}",
                "total_population": data["total_population"],
                "supply_rate": supply_rate,
                "supply_total": data["supply_all"],
                "sub_population": data["sub_population"],
                "region_population": region_population,
                "Allocation": allocation.tolist() if isinstance(allocation, np.ndarray) else allocation,
                "utility_slope": data["slope"],
                "Achieved Utility": float('nan'),
                "Total Utility": float('nan'),
                "Max Utility": data["max_utility"],
                "Gini Value": float('nan')
            })
        else:
            achieved_utility, total_achieved_utility = get_utility_value(sub_population, slope, allocation)
            gini_value = get_gini_value(sub_population, slope, allocation, max_utility)
            results.append({
                # 배분 결과가 하나일 경우 숫자 없이 시나리오 이름 생성
                "Scenario": allocation_type if len(allocations) == 1 else f"{allocation_type} {i + 1}",
                "total_population": data["total_population"],
                "supply_rate": supply_rate,
                "supply_total": data["supply_all"],
                "sub_population": data["sub_population"],
                "region_population": region_population,
                "Allocation": allocation.tolist() if isinstance(allocation, np.ndarray) else allocation,
                "utility_slope": data["slope"],
                "Achieved Utility": achieved_utility.tolist() if isinstance(achieved_utility, np.ndarray) else achieved_utility,
                "Total Utility": total_achieved_utility,
                "Max Utility": data["max_utility"],
                "Gini Value": gini_value
            })

def process_allocations_fair(allocations, epsilon, sub_population, slope, max_utility, supply_rate, data, results):
    #For fair allocation only
    """
    Process a list of allocations and append results to the results list.

    Args:
        allocations (list): List of allocation pairs.
        allocation_type (str): Type of allocation (e.g., 'Equal', 'Proportion').
        sub_population (list): Sub-population data.
        slope (float): Slope value for utility calculation.
        max_utility (float): Maximum utility value.
        results (list): List to store the results.
    """

    for i, allocation in enumerate(allocations):
        #print(allocation)
        achieved_utility, total_achieved_utility = get_utility_value(sub_population, slope, allocation)
        gini_value = get_gini_value(sub_population, slope, allocation, max_utility)
        results.append({
            "Scenario": "Fair", # 시나리오 이름 통일
            "total_population": data["total_population"],
            "supply_rate": supply_rate,
            "supply_total": data["supply_all"],
            "sub_population": data["sub_population"],
            "region_population": data["region_population"],
            "Allocation": allocation.tolist() if isinstance(allocation, np.ndarray) else allocation,
            "utility_slope": data["slope"],
            "Achieved Utility": achieved_utility.tolist() if isinstance(achieved_utility, np.ndarray) else achieved_utility,
            "Total Utility": total_achieved_utility,
            "Max Utility": data["max_utility"],
            "Gini Value": gini_value,
            "Epsilon": epsilon # Epsilon 값 별도 저장
        })

def get_experiment_condtitions():
    """conditions file에서 실험 조건을 가져옴"""
    n_all = np.array(conditions.populations)
    aged_ratio = np.array(conditions.aged_ratio)
    n_elders = n_all * aged_ratio
    n_adults = n_all * (1 - aged_ratio)
    total_population = sum(n_all)  # Total population across all regions
    slope_values = conditions.slope_values  # Slope values for utility calculation
    # Validate input lengths
    n_locations = len(n_elders)
    if len(n_adults) != n_locations or len(slope_values) != n_locations:
        raise ValueError("Input lengths must match the number of locations.")
    
    # Supply rate increments from 0 to 1
    n_supply_rate = conditions.n_supply_rate  # 등분 개수
    supply_rates = np.linspace(0, 0.5, n_supply_rate + 1)[0:]  # 0부터 1까지 n등분하고, 0 제외
    supply_rates = [round(x, 2) for x in supply_rates]  # 소수점 두 자리로 반올림
    # n = 10 -> [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    #epsilon value from 0 to 1 
    n_epsilon = conditions.n_epsilon  # 등분 개수
    epsilon_values = np.linspace(0, 0.5, n_epsilon + 1)[1:]  # 0부터 1까지 n등분하고, 0 제외
    epsilon_values = [round(x, 2) for x in epsilon_values] # 소수점 두 자리로 반올림
    # ex: n =20 -> epsilon_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    
    return {
        "n_all": n_all,
        "aged_ratio": aged_ratio,
        "n_elders": n_elders,
        "n_adults": n_adults,
        "supply_rates": supply_rates,
        "slope": slope_values,
        "epsilon_values": epsilon_values,
        "total_population": total_population,
    }

def define_one_case(n_elders, n_adults, slope_values, supply_rate):
    """
    Define the necessary data for the problem based on user input.

    Args:
        n_elders (list): Number of elders at each location.
        n_adults (list): Number of adults at each location.
        slope_values (list of lists): Slope values for [elders, adults] at each location.

    Returns:
        dict: Processed data for calculations.
    """
    # Validate input lengths
    n_locations = len(n_elders)
    if len(n_adults) != n_locations or len(slope_values) != n_locations:
        raise ValueError("Input lengths must match the number of locations.")

    data = {
        "location_id": list(range(n_locations)),
        "n_elders": n_elders,
        "n_adults": n_adults,
        "slope": slope_values,
        "supply_rate": supply_rate
    }

    # Calculate derived values
    supply_adults = supply_rate * sum(n_adults)
    supply_elders = supply_rate * sum(n_elders)
    supply_all = supply_adults + supply_elders
    total_population = sum(n_adults)+ sum(n_elders)
    region_population = np.array([n_elders[i]+ n_adults[i] for i in range(n_locations)])

    sub_population = np.array([[n_elders[i], n_adults[i]] for i in range(n_locations)])
    slope = np.array(slope_values)

    max_utility = np.array([np.sum(sub_population[i] * slope[i]) for i in range(n_locations)])

    return {
        "supply_rate": supply_rate,
        "total_population": total_population,
        "supply_adults": supply_adults,
        "supply_elders": supply_elders,
        "supply_all": supply_all,
        "sub_population": sub_population,
        "region_population": region_population,
        "slope": slope,
        "max_utility": max_utility
    }
    
def generate_allocations(n_all, supply_rate):
    num_regions = len(n_all)  # Determine the number of regions
    
    # Equal allocation: distribute equally among regions
    total_supply = sum(n_all) * supply_rate
    equal_allocation = [
        [total_supply / num_regions] * num_regions
    ]

    # Proportional allocation: distribute based on population proportions
    proportion_allocation = [
        [pop * supply_rate for pop in n_all]
    ]

    return equal_allocation, proportion_allocation

def run_allocation_scenarios(df_results=None):
    """모든 배분 시나리오(Equal, Proportion, Utility, Fair)를 실행하고 결과를 하나의 파일에 저장합니다."""
    exp_conditions = get_experiment_condtitions()
    results = []

    # 1. Equal, Proportion, Utility 시나리오 실행
    print("Running Equal, Proportion, and Utility scenarios...")
    for supply_rate in exp_conditions["supply_rates"]:
        data = define_one_case(exp_conditions["n_elders"], exp_conditions["n_adults"], exp_conditions["slope"], supply_rate)

        sub_population = data["sub_population"]
        slope = data["slope"]
        max_utility = data["max_utility"]
        region_population = data["region_population"]
        supply_all = data["supply_all"]

        # Equal & Proportion Allocation
        equal_allocation, proportion_allocation = generate_allocations(exp_conditions["n_all"], supply_rate)
        process_allocations(equal_allocation, "Equal", sub_population, slope, max_utility, supply_rate, data, results)
        process_allocations(proportion_allocation, "Proportion", sub_population, slope, max_utility, supply_rate, data, results)

        # Utility Maximized Allocation
        num_breaks = np.zeros(len(exp_conditions["n_all"]))
        for i in range(len(exp_conditions["n_all"])):
            num_breaks[i] = 2
        
        epsilon_for_utility = 1.0 # Epsilon=1은 효용 극대화와 동일
        x, _, _, _, _ = solve_problem_gurobi(epsilon_for_utility, supply_all, region_population, num_breaks, sub_population, slope, max_utility)
        
        allocation = np.zeros(len(exp_conditions["n_all"]))
        for i in range(len(exp_conditions["n_all"])):
            temp_x = 0
            for t in range(int(num_breaks[0])):
                temp_x += x.get((i, t), 0)
            allocation[i] = temp_x
        utility_allocation = [allocation]
        process_allocations(utility_allocation, "Utility", sub_population, slope, max_utility, supply_rate, data, results)

    # 2. Fair 시나리오 실행
    print("Running Fair scenarios...")
    for epsilon in exp_conditions["epsilon_values"]:
        for supply_rate in exp_conditions["supply_rates"]:
            data = define_one_case(exp_conditions["n_elders"], exp_conditions["n_adults"], exp_conditions["slope"], supply_rate)

            sub_population = data["sub_population"]
            slope = data["slope"]
            max_utility = data["max_utility"]
            region_population = data["region_population"]
            supply_all = data["supply_all"]

            num_breaks = np.zeros(len(exp_conditions["n_all"]))
            for i in range(len(exp_conditions["n_all"])):
                num_breaks[i] = 2

            x, _, _, _, _ = solve_problem_gurobi(epsilon, supply_all, region_population, num_breaks, sub_population, slope, max_utility)
            
            allocation = np.zeros(len(exp_conditions["n_all"]))
            for i in range(len(exp_conditions["n_all"])):
                temp_x = 0
                for t in range(int(num_breaks[0])):
                    temp_x += x.get((i, t), 0)
                allocation[i] = temp_x
            fair_allocation = [allocation]
            process_allocations_fair(fair_allocation, epsilon, sub_population, slope, max_utility, supply_rate, data, results)

    # 3. 모든 결과를 하나의 DataFrame으로 변환하고 저장
    df_results = pd.DataFrame(results)
    df_results.sort_values(by=["Scenario", "supply_rate", "Epsilon"], inplace=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, 'df_results_output.csv')
    df_results.to_csv(output_file_path, index=False)
    print(f"All allocation results saved to {output_file_path}")
