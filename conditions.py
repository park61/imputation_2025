import numpy as np

# 파라미터 설정
n_regions = 2
mask_durations_range = range(1, 6)  # 1부터 5까지 변화
mask_durations_fixed = 1# 1일 고정

populations = [800000, 400000]#인구수
default_aged_ratio = [0.5, 0.25]  # 고령층 비율
slope_values = [[5, 1], [3, 2]]  # Utility 계산을 위한 기울기 값

def get_aged_ratio_scenarios():
    """다양한 aged_ratio 시나리오를 생성합니다."""
    scenarios = []
    ratio_values = [round(r, 1) for r in np.arange(0.1, 1.1, 0.3)]
    # aged_ratio를 0.1부터 1.0까지 0.3 단위로 변경 (예: 0.1, 0.4, 0.7, 1.0)
    for r1 in ratio_values:
        for r2 in ratio_values:
            # 두 지역의 aged_ratio를 각각 독립적으로 설정
            scenarios.append([r1, r2])
    return scenarios

n_supply_rate = 10  # supply rates 등분 개수
n_epsilon = 20  # epsilon 등분 개수

contact_matrix = np.array([
        [1.0, 0.0],#지역 i는 지역 j의 인구 중 n%에 영향을 받음 
        [0.0, 1.0]
    ])#지역 간 영향력

# 파라미터 설정
params = [
    {'beta': 0.3, 'gamma': 1/6, 'sigma': 1/14, 'mu_na': 0.013, 'mu_a': 0.097, 'epsilon': 0.75, 'initial_infection_rate': 0.03},
    {'beta': 0.3, 'gamma': 1/6, 'sigma': 1/14, 'mu_na': 0.013, 'mu_a': 0.097, 'epsilon': 0.75, 'initial_infection_rate': 0.03},
]#0.013

supply_type='by_policy'#이거 제대로 작동안함...다른 방식을 넣기가 어려운데

# 고령층 우선 배분 파라미터
aged_priority_ratio = 1.0  # 고령층 우선비율 (0.5 ~ 1.0: 50% ~ 100%)
