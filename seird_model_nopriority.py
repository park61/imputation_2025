import supplyndemand as snd
import conditions as cnd
import numpy as np
from scipy.integrate import solve_ivp  # odeint 대신 solve_ivp 사용
import os
import pandas as pd

def multi_region_seir_model_with_mask_groups(y, t, params, n_regions, contact_matrix, mask_duration, mask_usages):
    dydt = []
    
    # Mask removal rate (1/duration gives us the rate of transition from masked to non-masked)
    mask_removal_rate = 1/mask_duration
    
    for i in range(n_regions):
        # Unpack state variables
        S_nam, S_na, E_nam, E_na, I_nam, I_na, R_nam, R_na, D_nam, D_na, \
        S_am, S_a, E_am, E_a, I_am, I_a, R_am, R_a, D_am, D_a = y[i * 20:(i + 1) * 20]
        
        # Total population in region i
        N = sum(y[i * 20:(i + 1) * 20])
        
        # Calculate current infected population for mask demand calculation
        current_infected = I_nam + I_na + I_am + I_a
        
        # Calculate dynamic mask usage rate
        
        if cnd.supply_type=='by_policy':
            mask_usage = mask_usages[i]
        else:
            mask_usage = snd.mask_use_rate(N, current_infected, t, cnd.supply_type, i)#region i
        
        # Unpack parameters
        beta, gamma, sigma, mu_na, mu_a, epsilon, initial_infection_ratio = params[i].values()
    
        #
        infection_force = 0
        # Calculate infection force for each subgroup
        for j in range(n_regions):
            # Unpack variables for region j
            _, _, _, _, I_nam_j, I_na_j, _, _, _, _, \
            _, _, _, _, I_am_j, I_a_j, _, _, _, _ = y[j * 20:(j + 1) * 20]
            N_j = sum(y[j * 20:(j + 1) * 20])

            infection_force += contact_matrix[i][j] * beta* (I_nam_j + I_am_j+I_na_j + I_a_j) / N_j#epsilon 위치 변경
            #inf_rate = R0 * prog_rate * rem_rate / (asym_inf * rem_rate + asym_inf * p_asym * prog_rate + (1 - p_asym) * prog_rate)
            #R0: beta, prog_rate: gamma (inc_rate가 1로 돼있음), rem_rate: sigma, asym_inf: , p_asym: 
            #asym.inf = 0.5, # relative infectiousness of asymptomatic/mildly symptomatic cases. Asymptomatic cases are half as infectious as symptomatic cases.
            #Asymptomatic or Presymptomatic 상태의 사람이 전염시키는 것은 Symptotic 상태 사람의 전염율의 50%라고 가정
            #p.asym: 전염력이 있는 사람 중 증상이 있는 사람의 비율 Pre 에서 asymptomatic으로 가는 비율
                        
        # Non-aged group with masks(reduced susceptibility due to masks))
        dS_nam = -S_nam * infection_force*(1-epsilon) - mask_removal_rate * S_nam + mask_usage * S_na
        dE_nam = S_nam * infection_force*(1-epsilon) - sigma * E_nam - mask_removal_rate * E_nam + mask_usage * E_na
        dI_nam = sigma * E_nam - gamma * I_nam - mu_na * I_nam - mask_removal_rate * I_nam + mask_usage * I_na
        dR_nam = gamma * I_nam - mask_removal_rate * R_nam + mask_usage * R_na
        dD_nam = mu_na * I_nam

        # Non-group without masks (full susceptibility)
        dS_na = -S_na * infection_force + mask_removal_rate * S_nam - mask_usage * S_na
        dE_na = S_na * infection_force - sigma * E_na + mask_removal_rate * E_nam - mask_usage * E_na
        dI_na = sigma * E_na - gamma * I_na - mu_na * I_na + mask_removal_rate * I_nam - mask_usage * I_na
        dR_na = gamma * I_na + mask_removal_rate * R_nam - mask_usage * R_na
        dD_na = mu_na * I_na

        # Aged group with masks (reduced susceptibility due to masks)
        dS_am = -S_am * infection_force*(1-epsilon) - mask_removal_rate * S_am + mask_usage * S_a
        dE_am = S_am * infection_force*(1-epsilon) - sigma * E_am - mask_removal_rate * E_am + mask_usage * E_a
        dI_am = sigma * E_am - gamma * I_am - mu_a * I_am - mask_removal_rate * I_am + mask_usage * I_a
        dR_am = gamma * I_am - mask_removal_rate * R_am + mask_usage * R_a
        dD_am = mu_a * I_am

        # Aged group without masks(full susceptibility)
        dS_a = -S_a * infection_force + mask_removal_rate * S_am - mask_usage * S_a
        dE_a = S_a * infection_force - sigma * E_a + mask_removal_rate * E_am - mask_usage * E_a
        dI_a = sigma * E_a - gamma * I_a - mu_a * I_a + mask_removal_rate * I_am - mask_usage * I_a
        dR_a = gamma * I_a + mask_removal_rate * R_am - mask_usage * R_a
        dD_a = mu_a * I_a

        # Fixed: Changed S_a to dS_a in the dydt array
        dydt.extend([
            dS_nam, dS_na, dE_nam, dE_na, dI_nam, dI_na, dR_nam, dR_na, dD_nam, dD_na,
            dS_am, dS_a, dE_am, dE_a, dI_am, dI_a, dR_am, dR_a, dD_am, dD_a  # Changed S_a to dS_a
        ])

    return dydt

def read_policies(filename):
    policies = {}
    df = pd.read_csv(filename)
    for _, row in df.iterrows():
        policies[row['policy_name']] = [
            int(row['region1_masks']),
            int(row['region2_masks']),
            float(row['aged_ratio_1']),
            float(row['aged_ratio_2'])
        ]
    return policies


def multi_region_seir_model_wrapper(t, y, params, n_regions, contact_matrix, mask_duration, mask_usages):
    """
    solve_ivp용 래퍼 함수 - 인자 순서가 (t, y)로 바뀜
    """
    return multi_region_seir_model_with_mask_groups(y, t, params, n_regions, contact_matrix, mask_duration, mask_usages)

def calculate_mortality(policy, policy_name, mask_duration):#mask_duration이 변할 수 있어서 인자로 받는다.
    region1_masks, region2_masks, aged_ratio_1, aged_ratio_2 = policy
    aged_ratios=[aged_ratio_1, aged_ratio_2]#노령층 비율

    # 초기 상태 설정
    start = []
    
    for i in range(cnd.n_regions):
        non_aged_pop = cnd.populations[i] * (1 - aged_ratios[i])
        aged_pop = cnd.populations[i] * aged_ratios[i]

        initial_mask_ratio = 0.1#ratio of initial mask users
        
        start.extend([
            # Non-aged group
            non_aged_pop * (1 - cnd.params[i]['initial_infection_rate']) * initial_mask_ratio,  # S_nam: m means mask, na means non-aged 
            non_aged_pop * (1 - cnd.params[i]['initial_infection_rate']) * (1 - initial_mask_ratio),  # S_na: no mask, a means aged
            non_aged_pop * cnd.params[i]['initial_infection_rate'] * initial_mask_ratio,  # E_nam
            non_aged_pop * cnd.params[i]['initial_infection_rate'] * (1 - initial_mask_ratio),  # E_na
            0, 0,  # I_nam, I_na
            0, 0,  # R_nam, R_na
            0, 0,  # D_nam, D_na
            
            # Aged group
            aged_pop * (1 - cnd.params[i]['initial_infection_rate']) * initial_mask_ratio,  # S_am
            aged_pop * (1 - cnd.params[i]['initial_infection_rate']) * (1 - initial_mask_ratio),  # S_a
            aged_pop * cnd.params[i]['initial_infection_rate'] * initial_mask_ratio,  # E_am
            aged_pop * cnd.params[i]['initial_infection_rate'] * (1 - initial_mask_ratio),  # E_a
            0, 0,  # I_am, I_a
            0, 0,  # R_am, R_a
            0, 0,  # D_am, D_a
        ])
    
    mask_usages=[m/p for m,p in zip ([region1_masks, region2_masks], cnd.populations)]#[region1_masks, region2_masks]는 정책에 따라 결정됨

    # solve_ivp 사용 
    sol = solve_ivp(
        fun=multi_region_seir_model_wrapper,
        t_span= (0, 500),
        y0=start,
        t_eval=np.linspace(0, 500, 5000) ,
        args=(cnd.params, cnd.n_regions, cnd.contact_matrix, mask_duration, mask_usages),
        method='LSODA',  # 옵션: 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA', 'RK45'
        rtol=1e-8,      # 상대 허용 오차
        atol=1e-10      # 절대 허용 오차
    )
    
    # 결과 추출 (solve_ivp는 전치된 형태로 결과를 반환)
    out = sol.y.T  # odeint와 같은 형태로 변환

    # 사망률 계산
    mortality_rates = {
    'region1_nonaged': (out[-1, 8] + out[-1, 9]) / (cnd.populations[0] * (1 - aged_ratios[0])),
    'region1_aged': (out[-1, 18] + out[-1, 19]) / (cnd.populations[0] * aged_ratios[0]),
    'region1': ((out[-1, 8] + out[-1, 9]) + (out[-1, 18] + out[-1, 19]))/cnd.populations[0],
    'region2_nonaged': (out[-1, 28] + out[-1, 29]) / (cnd.populations[1] * (1 - aged_ratios[1])),
    'region2_aged': (out[-1, 38] + out[-1, 39]) / (cnd.populations[1] * aged_ratios[1]),
    'region2': ((out[-1, 28] + out[-1, 29]) + (out[-1, 38] + out[-1, 39]))/cnd.populations[1],
    }
    
    # 총 사망률 계산 추가
    total_deaths = sum(out[-1, [8, 9, 18, 19,28,29,38,39]])
    total_population = sum(cnd.populations)
    mortality_rates['total'] = total_deaths / total_population
    return mortality_rates, out

def run_dynamic_experiment():
    
    """
    policy.csv 파일을 읽어 모든 정책에 대해 SEIRD 모델을 실행하고 결과를 CSV로 저장합니다.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, 'input')
    input_csv_path = os.path.join(input_dir, 'policy.csv')
    policies = read_policies(input_csv_path)
    results = {}

    #결과값 저장. Duration이 고정인 경우
    policy_results={}
    out_results={}
    for policy_name, policy in policies.items():
        mortality_rates,out = calculate_mortality(policy, policy_name, cnd.mask_durations_fixed)
        policy_results[cnd.mask_durations_fixed] = mortality_rates
        results[policy_name] = mortality_rates
        out_results[policy_name] = out

    output_dir = os.path.join(script_dir, 'output')
    output_csv_path = os.path.join(output_dir, 'policy_results_without_priority.csv')
    
    with open(output_csv_path, 'w') as f:
        # Write header
        f.write("Region1,Region2,Region1_NonAged,Region2_NonAged,Region1_Aged,Region2_Aged,Total_Mortality,policy,supply_rate,epsilon,R1_Masks,R2_Masks\n")
        
        # Write data for each policy
        for policy_name, mortality_rates in results.items():
            policy_parts = policy_name.split('_')
            
            # Get mask counts from the original policies dictionary
            region1_masks = policies[policy_name][0]
            region2_masks = policies[policy_name][1]
            
            policy_type=policy_parts[0]
            supply_rate=float(policy_parts[1])
            if len(policy_parts) == 3:
                epsilon=float(policy_parts[2])
            else:
                epsilon=""    
                
            f.write(f"{mortality_rates['region1']:.6f},"
                    f"{mortality_rates['region2']:.6f},"
                    f"{mortality_rates['region1_nonaged']:.6f},"
                    f"{mortality_rates['region2_nonaged']:.6f},"
                    f"{mortality_rates['region1_aged']:.6f},"
                    f"{mortality_rates['region2_aged']:.6f},"
                    f"{mortality_rates['total']:.6f},"
                    f"{policy_type},"
                    f"{supply_rate},"
                    f"{epsilon},"
                    f"{region1_masks},"
                    f"{region2_masks}\n")