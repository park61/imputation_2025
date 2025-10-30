import os
import pandas as pd
import conditions as cnd
import ast

def convert_df():
    """
    'df_results_output.csv' 파일을 읽어 'policy.csv' 형식으로 변환하고 저장합니다.
    """
    print("Converting allocation results to policy.txt for SEIRD model...")
    
    # 현재 스크립트의 디렉토리를 기준으로 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    input_dir = os.path.join(script_dir, 'input')
    
    # 입력/출력 파일 경로 생성
    csv_input_path = os.path.join(output_dir, 'df_results_output.csv')
    csv_output_path = os.path.join(input_dir, 'policy.csv')
    
    # output 디렉토리가 없으면 생성
    os.makedirs(input_dir, exist_ok=True)

    # CSV 파일 읽기
    df = pd.read_csv(csv_input_path)

    policy_lines = []
    aged_ratio_1, aged_ratio_2 = cnd.aged_ratio

    for _, row in df.iterrows():
        # 1. 정책 이름 생성
        if row['Scenario'] == 'Fair':
            policy_name = f"Fair_{row['supply_rate']}_{row['Epsilon']}"
        else:
            policy_name = f"{row['Scenario']}_{row['supply_rate']}"
            
        try:
            # 2. 마스크 할당량 파싱 및 정수 변환
            allocation_str = row['Allocation'].replace('np.float64', '')
            allocation = ast.literal_eval(allocation_str)
            region1_masks = int(allocation[0])
            region2_masks = int(allocation[1])
            
            # 3. 최종 라인 생성
            policy_lines.append({
                'policy_name': policy_name,
                'region1_masks': region1_masks,
                'region2_masks': region2_masks,
                'aged_ratio_1': aged_ratio_1,
                'aged_ratio_2': aged_ratio_2
            })
        except (ValueError, SyntaxError) as e:
            print(f"WARNING: Could not parse Allocation for policy {policy_name}: {row['Allocation']}. Skipping row. Error: {e}")

    # policy.csv 파일로 저장
    policy_df = pd.DataFrame(policy_lines)
    policy_df.to_csv(csv_output_path, index=False)
        
    print(f"Successfully created {csv_output_path}")