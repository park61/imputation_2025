import allocate_masks as alloc
import allocation_graphs as alloc_graph
import seird_model as seird
import seird_model_nopriority as seird_nopriority
import seird_graphs as seird_graph
import utils as utils
import conditions as cnd
import os

def run_single_scenario(drawingMode=False):
    """단일 시나리오(기본값)에 대해 전체 프로세스를 실행합니다."""
    # 1. 마스크 배분 시나리오 실행 및 결과 저장
    alloc.run_scenarios("All") # "All" 또는 "Fair"
    alloc_graph.draw_graphs(drawingMode)
    alloc_graph.draw_fair_graphs(drawingMode)

    # 2. 배분 결과를 SEIRD 모델 입력 형식으로 변환
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    input_dir = os.path.join(script_dir, 'input')
    results_csv = os.path.join(output_dir, 'df_results_output.csv')
    policy_csv = os.path.join(input_dir, 'policy.csv')
    utils.convert_df(results_csv, policy_csv, cnd.default_aged_ratio)

    # 3. 변환된 policy.csv를 사용하여 SEIRD 모델 실행
    seird.run_dynamic_experiment()
    seird_nopriority.run_dynamic_experiment()
    
    # 4. SEIRD 모델 결과 그래프 생성
    seird_graph.main_comparison_analysis(drawingMode)

def run_experiment_scenarios(drawingMode=False):
    """aged_ratio를 변경해가며 여러 실험 시나리오를 실행합니다."""
    aged_ratio_scenarios = cnd.get_aged_ratio_scenarios()
    
    for aged_ratio in aged_ratio_scenarios:
        print(f"\n===== Running experiment for aged_ratio: {aged_ratio} =====\n")
        
        # 파일명 생성을 위한 aged_ratio 문자열 (e.g., [0.1, 0.4] -> "01_04")
        ar_str = f"{str(aged_ratio[0]).replace('.', '')}_{str(aged_ratio[1]).replace('.', '')}"

        # 1. 마스크 배분 시나리오 실행
        # alloc.run_scenarios(mode="Fair", aged_ratio_override=aged_ratio)
        alloc.run_scenarios(mode="Fair", aged_ratio_override=aged_ratio)

        # 2. 배분 결과를 SEIRD 모델 입력 형식으로 변환
        # 각 시나리오 결과를 별도 파일로 관리
        results_csv = os.path.join('output', f'df_results_output_ar_{ar_str}.csv')
        policy_csv = os.path.join('output', f'policy_ar_{ar_str}.csv')
        utils.convert_df(results_csv, policy_csv, aged_ratio)

        # 3. 변환된 policy.csv를 사용하여 SEIRD 모델 실행
        seird.run_dynamic_experiment(policy_csv, os.path.join('output', f'policy_results_with_priority_ar_{ar_str}.csv'))
        seird_nopriority.run_dynamic_experiment(policy_csv, os.path.join('output', f'policy_results_without_priority_ar_{ar_str}.csv'))

        # 4. SEIRD 모델 결과 그래프 생성
        aged_priority_file = os.path.join('output', f'policy_results_with_priority_ar_{ar_str}.csv')
        non_priority_file = os.path.join('output', f'policy_results_without_priority_ar_{ar_str}.csv')
        
        # 그래프 파일명이 겹치지 않도록 aged_ratio 정보를 포함한 디렉토리 생성
        seird_graph.main_comparison_analysis(drawingMode, aged_priority_file, non_priority_file, aged_ratio)

if __name__ =="__main__":
    drawingMode = False  # 그래프 그릴지 여부 설정
    run_experiment_scenarios(drawingMode)
    # run_single_scenario(drawingMode) # 기존 방식대로 단일 실행이 필요할 경우 사용