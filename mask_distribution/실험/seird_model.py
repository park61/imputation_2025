import supplyndemand as snd
import conditions as cnd
import numpy as np
from scipy.integrate import solve_ivp
import os
import pandas as pd

# 속도 최적화 옵션
SAVE_INDIVIDUAL_CSV = True  # False로 설정하면 개별 파일 저장 안 함 (속도 향상)

def multi_region_seir_model_with_mask_groups(y, t, params, n_regions, contact_matrix, mask_duration, mask_usages_aged, mask_usages_nonaged):
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
        
        # Calculate dynamic mask usage rate - 이제 연령층별로 다름
        if cnd.supply_type=='by_policy':
            mask_usage_aged = mask_usages_aged[i]
            mask_usage_nonaged = mask_usages_nonaged[i]
        else:
            # 동적 계산의 경우도 연령층별로 분리 (나중에 구현)
            mask_usage_aged = snd.mask_use_rate(N, current_infected, t, cnd.supply_type, i)
            mask_usage_nonaged = snd.mask_use_rate(N, current_infected, t, cnd.supply_type, i)
        
        # Unpack parameters
        beta, gamma, sigma, mu_na, mu_a, epsilon, initial_infection_ratio = params[i].values()
    
        # Calculate infection force
        infection_force = 0
        for j in range(n_regions):
            # Unpack variables for region j
            _, _, _, _, I_nam_j, I_na_j, _, _, _, _, \
            _, _, _, _, I_am_j, I_a_j, _, _, _, _ = y[j * 20:(j + 1) * 20]
            N_j = sum(y[j * 20:(j + 1) * 20])

            infection_force += contact_matrix[i][j] * beta * (I_nam_j + I_am_j + I_na_j + I_a_j) / N_j
                        
        # Non-aged group with masks (reduced susceptibility due to masks)
        dS_nam = -S_nam * infection_force*(1-epsilon) - mask_removal_rate * S_nam + mask_usage_nonaged * S_na
        dE_nam = S_nam * infection_force*(1-epsilon) - sigma * E_nam - mask_removal_rate * E_nam + mask_usage_nonaged * E_na
        dI_nam = sigma * E_nam - gamma * I_nam - mu_na * I_nam - mask_removal_rate * I_nam + mask_usage_nonaged * I_na
        dR_nam = gamma * I_nam - mask_removal_rate * R_nam + mask_usage_nonaged * R_na
        dD_nam = mu_na * I_nam

        # Non-aged group without masks (full susceptibility)
        dS_na = -S_na * infection_force + mask_removal_rate * S_nam - mask_usage_nonaged * S_na
        dE_na = S_na * infection_force - sigma * E_na + mask_removal_rate * E_nam - mask_usage_nonaged * E_na
        dI_na = sigma * E_na - gamma * I_na - mu_na * I_na + mask_removal_rate * I_nam - mask_usage_nonaged * I_na
        dR_na = gamma * I_na + mask_removal_rate * R_nam - mask_usage_nonaged * R_na
        dD_na = mu_na * I_na

        # Aged group with masks (reduced susceptibility due to masks)
        dS_am = -S_am * infection_force*(1-epsilon) - mask_removal_rate * S_am + mask_usage_aged * S_a
        dE_am = S_am * infection_force*(1-epsilon) - sigma * E_am - mask_removal_rate * E_am + mask_usage_aged * E_a
        dI_am = sigma * E_am - gamma * I_am - mu_a * I_am - mask_removal_rate * I_am + mask_usage_aged * I_a
        dR_am = gamma * I_am - mask_removal_rate * R_am + mask_usage_aged * R_a
        dD_am = mu_a * I_am

        # Aged group without masks (full susceptibility)
        dS_a = -S_a * infection_force + mask_removal_rate * S_am - mask_usage_aged * S_a
        dE_a = S_a * infection_force - sigma * E_a + mask_removal_rate * E_am - mask_usage_aged * E_a
        dI_a = sigma * E_a - gamma * I_a - mu_a * I_a + mask_removal_rate * I_am - mask_usage_aged * I_a
        dR_a = gamma * I_a + mask_removal_rate * R_am - mask_usage_aged * R_a
        dD_a = mu_a * I_a

        dydt.extend([
            dS_nam, dS_na, dE_nam, dE_na, dI_nam, dI_na, dR_nam, dR_na, dD_nam, dD_na,
            dS_am, dS_a, dE_am, dE_a, dI_am, dI_a, dR_am, dR_a, dD_am, dD_a
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

def multi_region_seir_model_wrapper(t, y, params, n_regions, contact_matrix, mask_duration, mask_usages_aged, mask_usages_nonaged):
    """
    solve_ivp용 래퍼 함수 - 인자 순서가 (t, y)로 바뀜
    """
    return multi_region_seir_model_with_mask_groups(y, t, params, n_regions, contact_matrix, mask_duration, mask_usages_aged, mask_usages_nonaged)

def calculate_aged_priority_mask_allocation(region_masks, aged_pop, nonaged_pop, priority_ratio):
    """
    고령층 우선 마스크 배분 계산
    
    Args:
        region_masks: 해당 지역에 배분된 총 마스크 수
        aged_pop: 고령층 인구수
        nonaged_pop: 비고령층 인구수
        priority_ratio: 고령층 우선비율 (0.5 ~ 1.0)
    
    Returns:
        (aged_masks, nonaged_masks): 각 연령층에 배분된 마스크 수
    """
    # 고령층에 우선 배분할 마스크 수
    priority_masks = region_masks * priority_ratio
    
    # 고령층에 실제 배분되는 마스크 (고령층 인구수를 초과할 수 없음)
    aged_masks = min(priority_masks, aged_pop)
    
    # 남은 마스크를 비고령층에 배분
    nonaged_masks = region_masks - aged_masks
    
    return aged_masks, nonaged_masks

def calculate_mortality(policy, policy_name, mask_duration):
    region1_masks, region2_masks, aged_ratio_1, aged_ratio_2 = policy
    aged_ratios = [aged_ratio_1, aged_ratio_2]
    
    # 연령층별 인구 계산
    aged_pops = [pop * ratio for pop, ratio in zip(cnd.populations, aged_ratios)]
    nonaged_pops = [pop * (1 - ratio) for pop, ratio in zip(cnd.populations, aged_ratios)]
    
    # 고령층 우선 마스크 배분 계산
    mask_usages_aged = []
    mask_usages_nonaged = []
    
    for i, region_masks in enumerate([region1_masks, region2_masks]):
        aged_masks, nonaged_masks = calculate_aged_priority_mask_allocation(
            region_masks, aged_pops[i], nonaged_pops[i], cnd.aged_priority_ratio
        )
        
        # 연령층별 마스크 착용률 계산
        mask_usage_aged = aged_masks / aged_pops[i] if aged_pops[i] > 0 else 0
        mask_usage_nonaged = nonaged_masks / nonaged_pops[i] if nonaged_pops[i] > 0 else 0
        
        mask_usages_aged.append(mask_usage_aged)
        mask_usages_nonaged.append(mask_usage_nonaged)
    
    # 초기 상태 설정
    start = []
    
    for i in range(cnd.n_regions):
        non_aged_pop = nonaged_pops[i]
        aged_pop = aged_pops[i]

        initial_mask_ratio = 0.1  # ratio of initial mask users
        
        start.extend([
            # Non-aged group
            non_aged_pop * (1 - cnd.params[i]['initial_infection_rate']) * initial_mask_ratio,  # S_nam
            non_aged_pop * (1 - cnd.params[i]['initial_infection_rate']) * (1 - initial_mask_ratio),  # S_na
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
    
    # 시뮬레이션 시간
    t_span = (0, 500)
    t_eval = np.linspace(0, 500, 5000)
    
    # solve_ivp 사용 
    sol = solve_ivp(
        fun=multi_region_seir_model_wrapper,
        t_span=t_span,
        y0=start,
        t_eval=t_eval,
        args=(cnd.params, cnd.n_regions, cnd.contact_matrix, mask_duration, mask_usages_aged, mask_usages_nonaged),
        method='LSODA',
        rtol=1e-8,
        atol=1e-10
    )
    
    # 결과 추출
    out = sol.y.T
    
    # 사망률 계산
    mortality_rates = {
        'region1_nonaged': (out[-1, 8] + out[-1, 9]) / nonaged_pops[0],
        'region1_aged': (out[-1, 18] + out[-1, 19]) / aged_pops[0],
        'region1': ((out[-1, 8] + out[-1, 9]) + (out[-1, 18] + out[-1, 19])) / cnd.populations[0],
        'region2_nonaged': (out[-1, 28] + out[-1, 29]) / nonaged_pops[1],
        'region2_aged': (out[-1, 38] + out[-1, 39]) / aged_pops[1],
        'region2': ((out[-1, 28] + out[-1, 29]) + (out[-1, 38] + out[-1, 39])) / cnd.populations[1],
    }
    
    # 총 사망률 계산
    total_deaths = sum(out[-1, [8, 9, 18, 19, 28, 29, 38, 39]])
    total_population = sum(cnd.populations)
    mortality_rates['total'] = total_deaths / total_population
    
    return mortality_rates, out, mask_usages_aged, mask_usages_nonaged

def run_dynamic_experiment(policy_df=None):
    """
    policy.csv 파일 또는 policy DataFrame을 읽어 모든 정책에 대해 SEIRD 모델을 실행하고 결과를 CSV로 저장합니다.
    
    Args:
        policy_df: policy DataFrame (optional). 제공되지 않으면 파일에서 읽음.
    
    Returns:
        results_df: 결과 DataFrame (allocation 정보 포함)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, 'input')
    input_csv_path = os.path.join(input_dir, 'policy.csv')
    
    # policy_df가 제공되지 않으면 파일에서 읽기
    if policy_df is None:
        policies = read_policies(input_csv_path)
    else:
        # DataFrame에서 policies dict 생성 (calculate_mortality가 기대하는 리스트 형태)
        policies = {}
        for _, row in policy_df.iterrows():
            policy_name = row['policy_name']
            try:
                r1 = int(row['region1_masks'])
                r2 = int(row['region2_masks'])
                ar1 = float(row['aged_ratio_1'])
                ar2 = float(row['aged_ratio_2'])
            except Exception:
                # 문자열일 가능성 대비 재파싱
                r1 = int(float(row['region1_masks']))
                r2 = int(float(row['region2_masks']))
                ar1 = float(row['aged_ratio_1']) if not isinstance(row['aged_ratio_1'], str) else float(row['aged_ratio_1'])
                ar2 = float(row['aged_ratio_2']) if not isinstance(row['aged_ratio_2'], str) else float(row['aged_ratio_2'])
            policies[policy_name] = [r1, r2, ar1, ar2]
    
    results = {}

    # 결과값 저장. Duration이 고정인 경우
    policy_results = {}
    out_results = {}
    mask_usage_results = {}

    for policy_name, policy in policies.items():
        mortality_rates, out, mask_usages_aged, mask_usages_nonaged = calculate_mortality(policy, policy_name, cnd.mask_durations_fixed)
        policy_results[cnd.mask_durations_fixed] = mortality_rates
        results[policy_name] = mortality_rates
        out_results[policy_name] = out
        
        # 마스크 착용률 저장
        mask_usage_results[policy_name] = {
            'region1_aged': mask_usages_aged[0],
            'region1_nonaged': mask_usages_nonaged[0],
            'region2_aged': mask_usages_aged[1],
            'region2_nonaged': mask_usages_nonaged[1]
        }
    
    output_dir = os.path.join(script_dir, 'output')
    
    # 실험 조건을 파일명에 포함
    pop_str = f"pop{cnd.populations[0]}_{cnd.populations[1]}"
    aged_str = f"aged{cnd.aged_ratio[0]}_{cnd.aged_ratio[1]}"
    slope_str = f"slope{cnd.slope_values[0][0]}_{cnd.slope_values[0][1]}_{cnd.slope_values[1][0]}_{cnd.slope_values[1][1]}"
    exp_conditions = f"{pop_str}_{aged_str}_{slope_str}"
    
    # df_results_output 파일 읽기 (utility 정보 포함)
    df_results_path = os.path.join(output_dir, f'df_results_output_{exp_conditions}.csv')
    if not os.path.exists(df_results_path):
        df_results_path = os.path.join(output_dir, 'df_results_output.csv')
    
    # utility 정보를 딕셔너리로 저장
    utility_data = {}
    try:
        df_results = pd.read_csv(df_results_path)
        for _, row in df_results.iterrows():
            scenario = row['Scenario']
            supply_rate = row['supply_rate']
            epsilon = row['Epsilon'] if pd.notna(row['Epsilon']) else None
            
            # policy_name 형식 맞추기
            if scenario == 'Fair' and epsilon is not None:
                policy_key = f"{scenario}_{supply_rate}_{epsilon}"
            else:
                policy_key = f"{scenario}_{supply_rate}"
            
            # Achieved Utility는 리스트 형태로 저장되어 있으므로 파싱 필요
            achieved_utility_str = row['Achieved Utility']
            try:
                # "[value1, value2]" 형식을 파싱
                achieved_utility = eval(achieved_utility_str)
                total_achieved_utility = sum(achieved_utility)
            except:
                total_achieved_utility = None
            
            utility_data[policy_key] = {
                'achieved_utility': total_achieved_utility,
                'total_utility': row['Total Utility'],
                'max_utility_str': row['Max Utility'],  # 배열 형태
                'gini_value': row['Gini Value'] if pd.notna(row['Gini Value']) else None
            }
    except Exception as e:
        print(f"Warning: Could not read utility data from {df_results_path}: {e}")
        utility_data = {}
    
    # 개별 실험 조건 파일
    output_csv_path = os.path.join(output_dir, f'policy_results_with_priority_{exp_conditions}.csv')
    
    # 누적 마스터 파일
    master_csv_path = os.path.join(output_dir, 'policy_results_with_priority_accumulated.csv')
    
    # 마스터 파일이 존재하는지 확인
    master_file_exists = os.path.exists(master_csv_path)
    
    # 결과 데이터 수집
    results_data = []
    header = ("Region1,Region2,Region1_NonAged,Region2_NonAged,Region1_Aged,Region2_Aged,Total_Mortality,"
              "R1_Aged_MaskRate,R1_NonAged_MaskRate,R2_Aged_MaskRate,R2_NonAged_MaskRate,"
              "policy,supply_rate,epsilon,aged_priority_ratio,"
              "Achieved_Utility,Total_Utility,Gini_Value,"
              "pop1,pop2,aged_ratio1,aged_ratio2,slope_na1,slope_a1,slope_na2,slope_a2\n")
    
    for policy_name, mortality_rates in results.items():
        policy_parts = policy_name.split('_')
        policy_type = policy_parts[0]
        supply_rate = float(policy_parts[1])
        
        epsilon = ""
        if policy_type == 'Fair' and len(policy_parts) == 3:
            epsilon = float(policy_parts[2])
            
        mask_rates = mask_usage_results[policy_name]
        
        # utility 데이터 가져오기
        utility_info = utility_data.get(policy_name, {})
        achieved_utility = utility_info.get('achieved_utility', '')
        total_utility = utility_info.get('total_utility', '')
        gini_value = utility_info.get('gini_value', '')
        
        # 빈 값 처리
        achieved_utility_str = f"{achieved_utility:.2f}" if achieved_utility != '' and achieved_utility is not None else ""
        total_utility_str = f"{total_utility:.2f}" if total_utility != '' and total_utility is not None else ""
        gini_value_str = f"{gini_value:.6f}" if gini_value != '' and gini_value is not None else ""
        
        line = (
            f"{mortality_rates['region1']:.6f},"
            f"{mortality_rates['region2']:.6f},"
            f"{mortality_rates['region1_nonaged']:.6f},"
            f"{mortality_rates['region2_nonaged']:.6f},"
            f"{mortality_rates['region1_aged']:.6f},"
            f"{mortality_rates['region2_aged']:.6f},"
            f"{mortality_rates['total']:.6f},"
            f"{mask_rates['region1_aged']:.6f},"
            f"{mask_rates['region1_nonaged']:.6f},"
            f"{mask_rates['region2_aged']:.6f},"
            f"{mask_rates['region2_nonaged']:.6f},"
            f"{policy_type},{supply_rate},{epsilon},{cnd.aged_priority_ratio},"
            f"{achieved_utility_str},"
            f"{total_utility_str},"
            f"{gini_value_str},"
            f"{cnd.populations[0]},"
            f"{cnd.populations[1]},"
            f"{cnd.aged_ratio[0]},"
            f"{cnd.aged_ratio[1]},"
            f"{cnd.slope_values[0][0]},"
            f"{cnd.slope_values[0][1]},"
            f"{cnd.slope_values[1][0]},"
            f"{cnd.slope_values[1][1]}\n"
        )
        results_data.append(line)
    
    # 개별 실험 조건 파일 쓰기 (옵션)
    if SAVE_INDIVIDUAL_CSV:
        with open(output_csv_path, 'w') as f:
            # Write header (without experimental condition columns)
            f.write("Region1,Region2,Region1_NonAged,Region2_NonAged,Region1_Aged,Region2_Aged,Total_Mortality,"
                    "R1_Aged_MaskRate,R1_NonAged_MaskRate,R2_Aged_MaskRate,R2_NonAged_MaskRate,"
                    "policy,supply_rate,epsilon,aged_priority_ratio,Achieved_Utility,Total_Utility,Gini_Value\n")
            for line in results_data:
                # 실험 조건 컬럼 제외하고 쓰기 (utility 정보는 포함, 마지막 8개 실험조건 컬럼만 제외)
                f.write(','.join(line.split(',')[:18]) + '\n')
    
    # 누적 마스터 파일 쓰기 (append 모드)
    with open(master_csv_path, 'a') as f:
        # 파일이 새로 생성되는 경우에만 헤더 작성
        if not master_file_exists:
            f.write(header)
        # 결과 데이터 추가
        for line in results_data:
            f.write(line)
            
    if SAVE_INDIVIDUAL_CSV:
        print(f"SEIRD model results saved to {output_csv_path}")
    print(f"Results accumulated to {master_csv_path}")
    
    # DataFrame으로 변환하여 반환
    import io
    csv_content = header + ''.join(results_data)
    df_results_output = pd.read_csv(io.StringIO(csv_content))
    
    # 컬럼 순서 재배치: 실험조건 -> policy/supply/epsilon/gini -> 나머지
    condition_cols = ['pop1', 'pop2', 'aged_ratio1', 'aged_ratio2', 'slope_na1', 'slope_a1', 'slope_na2', 'slope_a2']
    key_cols = ['policy', 'supply_rate', 'epsilon', 'Gini_Value']
    mortality_cols = ['Region1', 'Region2', 'Region1_NonAged', 'Region2_NonAged', 'Region1_Aged', 'Region2_Aged', 'Total_Mortality']
    mask_cols = ['R1_Aged_MaskRate', 'R1_NonAged_MaskRate', 'R2_Aged_MaskRate', 'R2_NonAged_MaskRate']
    utility_cols = ['aged_priority_ratio', 'Achieved_Utility', 'Total_Utility']
    
    # 재배치된 순서
    ordered_cols = condition_cols + key_cols + mortality_cols + mask_cols + utility_cols
    # 존재하는 컬럼만 선택
    available_cols = [col for col in ordered_cols if col in df_results_output.columns]
    df_results_output = df_results_output[available_cols]
    
    return df_results_output