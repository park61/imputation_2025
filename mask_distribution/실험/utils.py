import os
import json
import pandas as pd
import conditions as cnd
import ast

def convert_df(df_allocation=None):
    """
    'df_results_output.csv' 파일을 읽어 'policy.csv' 형식으로 변환하고 저장합니다.
    df_allocation이 제공되면 파일 대신 이를 사용합니다.
    
    Args:
        df_allocation: allocation 결과 DataFrame (optional)
    
    Returns:
        policy_df: 변환된 policy DataFrame
    """
    print("Converting allocation results to policy.txt for SEIRD model...")
    
    # 현재 스크립트의 디렉토리를 기준으로 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    input_dir = os.path.join(script_dir, 'input')
    
    # 실험 조건을 파일명에 포함
    pop_str = f"pop{cnd.populations[0]}_{cnd.populations[1]}"
    aged_str = f"aged{cnd.aged_ratio[0]}_{cnd.aged_ratio[1]}"
    slope_str = f"slope{cnd.slope_values[0][0]}_{cnd.slope_values[0][1]}_{cnd.slope_values[1][0]}_{cnd.slope_values[1][1]}"
    exp_conditions_str = f"{pop_str}_{aged_str}_{slope_str}"
    
    # DataFrame이 제공되지 않은 경우 파일에서 읽기
    if df_allocation is None:
        # 입력/출력 파일 경로 생성
        csv_input_path = os.path.join(output_dir, f'df_results_output_{exp_conditions_str}.csv')
        
        # 파일이 없으면 기본 파일명으로 시도
        if not os.path.exists(csv_input_path):
            print(f"Warning: {csv_input_path} not found. Trying default filename...")
            csv_input_path = os.path.join(output_dir, 'df_results_output.csv')
        
        # CSV 파일 읽기
        df = pd.read_csv(csv_input_path)
    else:
        df = df_allocation
    
    csv_output_path = os.path.join(input_dir, 'policy.csv')
    
    # output 디렉토리가 없으면 생성
    os.makedirs(input_dir, exist_ok=True)

    policy_lines = []
    aged_ratio_1, aged_ratio_2 = cnd.aged_ratio

    for _, row in df.iterrows():
        # 1. 정책 이름 생성
        if row['Scenario'] == 'Fair':
            policy_name = f"Fair_{row['supply_rate']}_{row['Epsilon']}"
        else:
            policy_name = f"{row['Scenario']}_{row['supply_rate']}"
            
        try:
            # 2. 마스크 할당량 파싱 및 정수 변환 (list/tuple/str 모두 허용)
            allocation_val = row['Allocation']

            if isinstance(allocation_val, (list, tuple)):
                allocation = list(allocation_val)
            elif isinstance(allocation_val, str):
                # 문자열이면 우선 literal_eval 시도, 실패 시 json.loads 시도
                parsed = None
                try:
                    parsed = ast.literal_eval(allocation_val)
                except Exception:
                    try:
                        parsed = json.loads(allocation_val)
                    except Exception:
                        parsed = None
                if parsed is None:
                    raise ValueError(f"Could not parse Allocation string: {allocation_val}")
                allocation = list(parsed)
            else:
                raise TypeError(f"Unsupported Allocation type: {type(allocation_val)}")

            # 값 보정 및 정수 변환
            region1_masks = int(round(float(allocation[0])))
            region2_masks = int(round(float(allocation[1])))
            
            # 3. 최종 라인 생성
            policy_lines.append({
                'policy_name': policy_name,
                'region1_masks': region1_masks,
                'region2_masks': region2_masks,
                'aged_ratio_1': aged_ratio_1,
                'aged_ratio_2': aged_ratio_2
            })
        except (ValueError, SyntaxError, TypeError) as e:
            print(f"WARNING: Could not parse Allocation for policy {policy_name}: {row['Allocation']}. Skipping row. Error: {e}")

    # policy.csv 파일로 저장
    policy_df = pd.DataFrame(policy_lines)
    policy_df.to_csv(csv_output_path, index=False)
        
    print(f"Successfully created {csv_output_path}")
    
    # DataFrame 반환
    return policy_df