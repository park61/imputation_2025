import allocate_masks as alloc
import allocation_graphs as alloc_graph
import seird_model as seird
import seird_model_nopriority as seird_nopriority
import seird_graphs as seird_graph
import utils as utils

if __name__ =="__main__":
    # # 1. 마스크 배분 시나리오 실행 및 결과 저장
    alloc.run_allocation_scenarios()
    alloc_graph.draw_graphs()
    #alloc_graph.draw_fair_graphs()

    # # 2. 배분 결과를 SEIRD 모델 입력 형식으로 변환
    utils.convert_df()

    # 3. 변환된 policy.txt를 사용하여 SEIRD 모델 실행
    seird.run_dynamic_experiment()
    seird_nopriority.run_dynamic_experiment()
    seird_graph.main_comparison_analysis()