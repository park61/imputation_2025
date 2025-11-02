import os
import pandas as pd
import conditions as cnd
import ast

def convert_df(input_csv_path, output_csv_path, aged_ratio):
    """
    지정된 결과 파일을 읽어 'policy.csv' 형식으로 변환하고 저장합니다.
    
    Args:
        input_csv_path (str): 읽어올 allocation 결과 CSV 파일 경로.
        output_csv_path (str): 저장할 policy CSV 파일 경로.
        aged_ratio (list): 현재 시나리오의 고령층 비율.
    """
    print(f"Converting {input_csv_path} to {output_csv_path} for SEIRD model...")

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # CSV 파일 읽기
    df = pd.read_csv(input_csv_path)

    policy_lines = []
    aged_ratio_1, aged_ratio_2 = aged_ratio

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
    policy_df.to_csv(output_csv_path, index=False)
        
    print(f"Successfully created {output_csv_path}")