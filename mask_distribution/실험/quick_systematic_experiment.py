"""
Quick Systematic Experiment Script (Optimized)
빠른 실험을 위한 최적화 버전 - 속도 향상 + 패턴 분석
"""

import os
import sys
import pandas as pd
import numpy as np
from itertools import product
import conditions as cnd
import allocate_masks as alloc
import utils
import seird_model as seird
import seird_model_nopriority as seird_no

# 속도 향상을 위한 설정
SAVE_INDIVIDUAL_FILES = False  # 개별 파일 저장 안 함
RUN_NOPRIORITY_MODEL = False   # Without priority 모델 생략
BATCH_SAVE_INTERVAL = 50       # 배치 저장 간격

def backup_conditions():
    """현재 conditions.py 설정을 백업"""
    return {
        'populations': cnd.populations.copy(),
        'aged_ratio': cnd.aged_ratio.copy(),
        'slope_values': [slope.copy() for slope in cnd.slope_values]
    }

def restore_conditions(backup):
    """conditions.py 설정을 복원"""
    cnd.populations = backup['populations']
    cnd.aged_ratio = backup['aged_ratio']
    cnd.slope_values = backup['slope_values']

def update_conditions(pop_r1, pop_r2, aged_r1, aged_r2, slope_aged_r1, slope_nonaged_r1, slope_aged_r2, slope_nonaged_r2):
    """conditions.py의 설정을 업데이트"""
    cnd.populations = [pop_r1, pop_r2]
    cnd.aged_ratio = [aged_r1, aged_r2]
    cnd.slope_values = [[slope_aged_r1, slope_nonaged_r1], [slope_aged_r2, slope_nonaged_r2]]

def analyze_epsilon_mortality_trend(csv_file, supply_rate=0.1):
    """Fair 정책에서 epsilon 증가에 따른 mortality 변화 분석"""
    try:
        df = pd.read_csv(csv_file)
        
        # Fair 정책, 특정 supply_rate 필터링
        df['policy_type'] = df['policy'].apply(lambda x: str(x).split('_')[0])
        fair_df = df[
            (df['policy_type'] == 'Fair') & 
            (df['supply_rate'] == supply_rate)
        ].copy()
        
        if fair_df.empty or len(fair_df) < 2:
            return None
        
        fair_df['epsilon'] = pd.to_numeric(fair_df['epsilon'], errors='coerce')
        fair_df.dropna(subset=['epsilon', 'Total_Mortality'], inplace=True)
        fair_df.sort_values('epsilon', inplace=True)
        
        if len(fair_df) < 2:
            return None
        
        epsilons = fair_df['epsilon'].values
        mortalities = fair_df['Total_Mortality'].values
        
        # 선형 회귀
        coefficients = np.polyfit(epsilons, mortalities, 1)
        slope = coefficients[0]
        intercept = coefficients[1]
        
        # R^2 계산
        y_pred = slope * epsilons + intercept
        ss_res = np.sum((mortalities - y_pred) ** 2)
        ss_tot = np.sum((mortalities - np.mean(mortalities)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'is_decreasing': slope < 0,
            'slope': slope,
            'r_squared': r_squared,
            'epsilon_range': (epsilons.min(), epsilons.max()),
            'mortality_range': (mortalities.min(), mortalities.max()),
            'data_points': len(fair_df)
        }
    except Exception as e:
        print(f"Error: {e}")
        return None

def run_single_experiment(config, verbose=True):
    """단일 실험 실행 (최적화 버전)"""
    pop_r1, pop_r2 = config['populations']
    aged_r1, aged_r2 = config['aged_ratio']
    slope_aged_r1, slope_nonaged_r1, slope_aged_r2, slope_nonaged_r2 = config['slopes']
    
    if verbose:
        print(f"\nPop:[{pop_r1},{pop_r2}] Aged:[{aged_r1},{aged_r2}] "
              f"Slope:R1({slope_aged_r1},{slope_nonaged_r1}) R2({slope_aged_r2},{slope_nonaged_r2})", end=" ")
    
    update_conditions(pop_r1, pop_r2, aged_r1, aged_r2, 
                     slope_aged_r1, slope_nonaged_r1, 
                     slope_aged_r2, slope_nonaged_r2)
    
    try:
        # 마스크 배분 실행
        alloc.run_allocation_scenarios()
        utils.convert_df()
        
        # With priority 모델만 실행 (속도 향상)
        seird.run_dynamic_experiment()
        
        # Without priority 모델은 옵션
        if RUN_NOPRIORITY_MODEL:
            seird_no.run_dynamic_experiment()
        
        # 누적 파일에서 직접 읽기 (개별 파일 불필요)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'output')
        
        # 누적 파일에서 현재 조건의 데이터만 추출
        accumulated_file = os.path.join(output_dir, 'policy_results_with_priority_accumulated_1029.csv')
        
        if os.path.exists(accumulated_file):
            df_all = pd.read_csv(accumulated_file)
            # 현재 실험 조건과 일치하는 행만 필터링
            # 주의: slope_values는 [[aged, nonaged], [aged, nonaged]] 구조
            # 누적 파일의 slope_na1은 slope_values[0][0] (aged), slope_a1은 slope_values[0][1] (nonaged)
            current_data = df_all[
                (df_all['pop1'] == pop_r1) &
                (df_all['pop2'] == pop_r2) &
                (df_all['aged_ratio1'] == aged_r1) &
                (df_all['aged_ratio2'] == aged_r2) &
                (df_all['slope_na1'] == slope_aged_r1) &
                (df_all['slope_a1'] == slope_nonaged_r1) &
                (df_all['slope_na2'] == slope_aged_r2) &
                (df_all['slope_a2'] == slope_nonaged_r2)
            ]
            
            # Fair 정책, supply_rate=0.1 필터링
            fair_df = current_data[
                (current_data['policy'] == 'Fair') &
                (current_data['supply_rate'] == 0.1)
            ].copy()
            
            if not fair_df.empty and len(fair_df) >= 2:
                fair_df['epsilon'] = pd.to_numeric(fair_df['epsilon'], errors='coerce')
                fair_df.dropna(subset=['epsilon', 'Total_Mortality'], inplace=True)
                fair_df.sort_values('epsilon', inplace=True)
                
                if len(fair_df) >= 2:
                    epsilons = fair_df['epsilon'].values
                    mortalities = fair_df['Total_Mortality'].values
                    
                    # 선형 회귀
                    coefficients = np.polyfit(epsilons, mortalities, 1)
                    slope = coefficients[0]
                    intercept = coefficients[1]
                    
                    # R^2 계산
                    y_pred = slope * epsilons + intercept
                    ss_res = np.sum((mortalities - y_pred) ** 2)
                    ss_tot = np.sum((mortalities - np.mean(mortalities)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    result = {
                        'pop_r1': pop_r1,
                        'pop_r2': pop_r2,
                        'aged_r1': aged_r1,
                        'aged_r2': aged_r2,
                        'slope_aged_r1': slope_aged_r1,
                        'slope_nonaged_r1': slope_nonaged_r1,
                        'slope_aged_r2': slope_aged_r2,
                        'slope_nonaged_r2': slope_nonaged_r2,
                        'is_decreasing': slope < 0,
                        'trend_slope': slope,
                        'r_squared': r_squared,
                        'mortality_change': mortalities.max() - mortalities.min(),
                        'mortality_start': mortalities[0] if len(mortalities) > 0 else None,
                        'mortality_end': mortalities[-1] if len(mortalities) > 0 else None,
                    }
                    
                    if verbose:
                        status = "✓" if slope < 0 else "✗"
                        print(f"{status} slope={slope:.8f} R²={r_squared:.3f}")
                    
                    return result
        
        if verbose:
            print("✗ No data")
        return None
            
    except Exception as e:
        if verbose:
            print(f"✗ Error: {e}")
        return None

def generate_focused_configs():
    """
    집중 실험 조건 생성 (확장 버전)
    총 인구 1,000,000명 고정
    - population_ratios: 0.3-0.7
    - aged_ratios: 0.1-0.5
    - slope_combinations: 1-5
    """
    total_population = 1000000
    
    # 인구 분포: 0.3-0.7 범위 (0.1 간격)
    population_ratios = [
        (0.3, 0.7),
        (0.4, 0.6),
        (0.5, 0.5),
        (0.6, 0.4),
        (0.7, 0.3),
    ]
    
    # 고령층 비율: 0.1-0.5 범위 (0.1 간격)
    aged_ratios = [
        (0.2, 0.1),
        (0.3, 0.2),
        (0.5, 0.3),
        (0.6, 0.4),
    ]
    
    # Slope 조합: 1-5 범위 (aged, non-aged)
    # aged slope >= non-aged slope 조건 만족
    slope_combinations = [        
        (5, 2),
        (5, 3),
        (4, 1),
        (4, 2),
        (3, 1),
        (3, 2),
        (2, 1),
        (2, 2),
        (1, 1),
    ]
    
    configs = []
    for pop_ratio, aged_ratio, slope_r1, slope_r2 in product(
        population_ratios, aged_ratios, slope_combinations, slope_combinations
    ):
        pop_r1 = int(total_population * pop_ratio[0])
        pop_r2 = int(total_population * pop_ratio[1])
        aged_r1, aged_r2 = aged_ratio
        slope_aged_r1, slope_nonaged_r1 = slope_r1
        slope_aged_r2, slope_nonaged_r2 = slope_r2
        
        # aged slope > non-aged slope 확인 (이미 slope_combinations에서 보장)
        configs.append({
            'populations': (pop_r1, pop_r2),
            'aged_ratio': (aged_r1, aged_r2),
            'slopes': (slope_aged_r1, slope_nonaged_r1, slope_aged_r2, slope_nonaged_r2)
        })
    
    return configs

def define_focused_configs():
    
    configs = []

    # 1
    configs.append({'populations': (400000, 600000), 'aged_ratio': (0.3, 0.2), 'slopes': (4, 2, 5, 3)})
    # 2
    configs.append({'populations': (500000, 500000), 'aged_ratio': (0.3, 0.2), 'slopes': (3, 1, 5, 2)})
    # 3
    configs.append({'populations': (600000, 400000), 'aged_ratio': (0.4, 0.3), 'slopes': (5, 2, 4, 2)})
    # 4
    configs.append({'populations': (400000, 600000), 'aged_ratio': (0.5, 0.3), 'slopes': (4, 1, 5, 2)})
    # 5
    configs.append({'populations': (500000, 500000), 'aged_ratio': (0.3, 0.2), 'slopes': (5, 3, 4, 2)})
    # 6
    configs.append({'populations': (450000, 550000), 'aged_ratio': (0.4, 0.3), 'slopes': (4, 2, 5, 3)})
    # 7
    configs.append({'populations': (550000, 450000), 'aged_ratio': (0.5, 0.3), 'slopes': (5, 3, 3, 1)})
    # 8
    configs.append({'populations': (600000, 400000), 'aged_ratio': (0.4, 0.2), 'slopes': (5, 2, 3, 1)})
    # 9
    configs.append({'populations': (500000, 500000), 'aged_ratio': (0.5, 0.3), 'slopes': (4, 2, 3, 1)})
    # 10
    configs.append({'populations': (400000, 600000), 'aged_ratio': (0.3, 0.2), 'slopes': (3, 1, 4, 2)})
    # 11
    configs.append({'populations': (450000, 550000), 'aged_ratio': (0.4, 0.3), 'slopes': (3, 1, 5, 2)})
    # 12
    configs.append({'populations': (550000, 450000), 'aged_ratio': (0.5, 0.3), 'slopes': (5, 3, 4, 2)})
    # 13
    configs.append({'populations': (500000, 500000), 'aged_ratio': (0.4, 0.3), 'slopes': (4, 2, 3, 1)})
    # 14
    configs.append({'populations': (600000, 400000), 'aged_ratio': (0.3, 0.2), 'slopes': (5, 2, 4, 2)})
    # 15
    configs.append({'populations': (400000, 600000), 'aged_ratio': (0.4, 0.3), 'slopes': (3, 1, 5, 3)})
    # 16
    configs.append({'populations': (500000, 500000), 'aged_ratio': (0.3, 0.2), 'slopes': (5, 3, 3, 1)})
    # 17
    configs.append({'populations': (450000, 550000), 'aged_ratio': (0.4, 0.3), 'slopes': (4, 2, 5, 2)})
    # 18
    configs.append({'populations': (550000, 450000), 'aged_ratio': (0.5, 0.3), 'slopes': (5, 3, 4, 1)})
    # 19
    configs.append({'populations': (600000, 400000), 'aged_ratio': (0.4, 0.2), 'slopes': (4, 2, 3, 1)})
    # 20
    configs.append({'populations': (400000, 600000), 'aged_ratio': (0.3, 0.2), 'slopes': (3, 1, 4, 1)})
    # 21
    configs.append({'populations': (450000, 550000), 'aged_ratio': (0.4, 0.3), 'slopes': (5, 2, 4, 2)})
    # 22
    configs.append({'populations': (550000, 450000), 'aged_ratio': (0.5, 0.3), 'slopes': (4, 1, 3, 1)})
    # 23
    configs.append({'populations': (500000, 500000), 'aged_ratio': (0.4, 0.3), 'slopes': (5, 2, 3, 1)})
    # 24
    configs.append({'populations': (600000, 400000), 'aged_ratio': (0.5, 0.3), 'slopes': (5, 3, 4, 2)})
    # 25
    configs.append({'populations': (400000, 600000), 'aged_ratio': (0.4, 0.2), 'slopes': (3, 1, 5, 2)})
    # 26
    configs.append({'populations': (450000, 550000), 'aged_ratio': (0.4, 0.3), 'slopes': (4, 2, 5, 3)})
    # 27
    configs.append({'populations': (550000, 450000), 'aged_ratio': (0.5, 0.3), 'slopes': (5, 3, 3, 1)})
    # 28
    configs.append({'populations': (600000, 400000), 'aged_ratio': (0.4, 0.3), 'slopes': (5, 2, 4, 2)})
    # 29
    configs.append({'populations': (500000, 500000), 'aged_ratio': (0.3, 0.2), 'slopes': (4, 1, 5, 2)})
    # 30
    configs.append({'populations': (400000, 600000), 'aged_ratio': (0.5, 0.3), 'slopes': (4, 2, 5, 3)})

    
    
    return configs


def analyze_patterns(supply_rate: float | None = None):
    """누적 파일에서 Fair 정책만 읽어 조건별 감소 패턴 분석 및 요약 산출
    - supply_rate가 지정되면 해당 값으로 필터링하고, 조건 키에도 포함
    - 지정하지 않으면 모든 supply_rate를 합쳐서 조건별 추세 계산
    """
    print("\n" + "="*80)
    print("PATTERN ANALYSIS")
    print("="*80)

    # 누적 파일 로드
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    accumulated_file = os.path.join(output_dir, 'policy_results_with_priority_accumulated_1029.csv')

    if not os.path.exists(accumulated_file):
        print(f"Not found: {accumulated_file}")
        return {}

    df_all = pd.read_csv(accumulated_file)

    # Fair 정책만 사용, epsilon/Total_Mortality 정수/문자 섞임 방지
    fair = df_all[df_all['policy'] == 'Fair'].copy()
    if supply_rate is not None:
        fair = fair[fair['supply_rate'] == supply_rate]
        print(f"Filter: supply_rate = {supply_rate}")
    if fair.empty:
        print("No Fair policy rows found.")
        return {}

    fair['epsilon'] = pd.to_numeric(fair['epsilon'], errors='coerce')
    fair['Total_Mortality'] = pd.to_numeric(fair['Total_Mortality'], errors='coerce')
    fair.dropna(subset=['epsilon', 'Total_Mortality'], inplace=True)

    # 조건 키 정의 (실험 조건의 유일성 판단 기준)
    # supply_rate를 항상 조건에 포함하여 각 supply_rate별로 독립 분석
    condition_keys = ['pop1','pop2','aged_ratio1','aged_ratio2','slope_na1','slope_a1','slope_na2','slope_a2']
    if supply_rate is None:
        # supply_rate 미지정 시, 모든 supply_rate를 조건으로 구분
        condition_keys = condition_keys + ['supply_rate']
    # else: 지정된 supply_rate만 필터링되어 있으므로 키에 추가 불필요

    # 총 고려된 조건(=Fair 데이터가 존재하는 유니크 조건) 집계
    total_conditions = fair[condition_keys].drop_duplicates().shape[0]

    # 조건별로 epsilon-사망률 추세 계산 (2개 이상 점 필요)
    rows = []
    for cond_vals, grp in fair.groupby(condition_keys):
        grp_sorted = grp.sort_values('epsilon')
        if grp_sorted['epsilon'].nunique() < 2:
            continue

        eps = grp_sorted['epsilon'].values
        morts = grp_sorted['Total_Mortality'].values

        # 선형 회귀 (epsilon -> mortality)
        try:
            coeff = np.polyfit(eps, morts, 1)
            slope = float(coeff[0])
            intercept = float(coeff[1])
            y_pred = slope * eps + intercept
            ss_res = float(np.sum((morts - y_pred) ** 2))
            ss_tot = float(np.sum((morts - np.mean(morts)) ** 2))
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        except Exception:
            # 회귀 실패 시 건너뜀
            continue

        # cond_vals may include supply_rate depending on condition_keys
        vals = cond_vals if isinstance(cond_vals, tuple) else (cond_vals,)
        m = dict(zip(condition_keys, vals))
        pop1, pop2 = m['pop1'], m['pop2']
        ar1, ar2 = m['aged_ratio1'], m['aged_ratio2']
        s_na1, s_a1, s_na2, s_a2 = m['slope_na1'], m['slope_a1'], m['slope_na2'], m['slope_a2']
        sr = m.get('supply_rate', supply_rate)  # supply_rate from condition or parameter

        # 주의: 누적 파일의 slope_naX가 aged, slope_aX가 non-aged로 저장됨
        row = {
            'pop_r1': int(pop1), 'pop_r2': int(pop2),
            'aged_r1': float(ar1), 'aged_r2': float(ar2),
            'slope_aged_r1': int(s_na1), 'slope_nonaged_r1': int(s_a1),
            'slope_aged_r2': int(s_na2), 'slope_nonaged_r2': int(s_a2),
            'supply_rate': float(sr) if sr is not None else None,
            'trend_slope': slope,
            'r_squared': r_squared,
            'mortality_change': float(morts[-1] - morts[0]),
            'is_decreasing': slope < 0,
        }
        rows.append(row)

    if not rows:
        print("No analyzable conditions (need >= 2 epsilon points per condition).")
        return {}

    df = pd.DataFrame(rows)

    analyzed_conditions = len(df)
    skipped_conditions = total_conditions - analyzed_conditions

    print(f"Total Fair conditions found: {total_conditions}")
    print(f"Analyzed conditions (>=2 epsilon points): {analyzed_conditions}")
    print(f"Skipped (insufficient points): {skipped_conditions}")

    patterns = {}
    patterns['meta'] = {
        'total_conditions': total_conditions,
        'analyzed_conditions': analyzed_conditions,
        'skipped_conditions': skipped_conditions,
    }

    # 1. 인구 분포 패턴
    print("\n1. POPULATION DISTRIBUTION PATTERN")
    print("-" * 80)
    df['pop_ratio'] = df.apply(lambda x: f"{x['pop_r1']/10000:.0f}:{x['pop_r2']/10000:.0f}", axis=1)
    pop_analysis = df.groupby('pop_ratio').agg({
        'is_decreasing': ['sum', 'count', 'mean'],
        'trend_slope': 'mean'
    }).round(4)
    pop_analysis.columns = ['Decreasing_Count', 'Total', 'Success_Rate', 'Avg_Slope']
    pop_analysis = pop_analysis.sort_values('Success_Rate', ascending=False)
    print(pop_analysis)
    patterns['population'] = pop_analysis

    # 2. 고령층 비율 패턴
    print("\n2. AGED RATIO PATTERN")
    print("-" * 80)
    df['aged_combo'] = df.apply(lambda x: f"R1:{x['aged_r1']},R2:{x['aged_r2']}", axis=1)
    aged_analysis = df.groupby('aged_combo').agg({
        'is_decreasing': ['sum', 'count', 'mean'],
        'trend_slope': 'mean'
    }).round(4)
    aged_analysis.columns = ['Decreasing_Count', 'Total', 'Success_Rate', 'Avg_Slope']
    aged_analysis = aged_analysis.sort_values('Success_Rate', ascending=False)
    print(aged_analysis)
    patterns['aged_ratio'] = aged_analysis

    # 3. Slope 차이 패턴
    print("\n3. SLOPE DIFFERENCE PATTERN")
    print("-" * 80)
    df['slope_diff_r1'] = df['slope_aged_r1'] - df['slope_nonaged_r1']
    df['slope_diff_r2'] = df['slope_aged_r2'] - df['slope_nonaged_r2']
    df['avg_slope_diff'] = (df['slope_diff_r1'] + df['slope_diff_r2']) / 2

    slope_bins = [0, 1.5, 2.5, 5]
    slope_labels = ['Small(≤1.5)', 'Medium(1.5-2.5)', 'Large(>2.5)']
    df['slope_diff_category'] = pd.cut(df['avg_slope_diff'], bins=slope_bins, labels=slope_labels)

    slope_analysis = df.groupby('slope_diff_category').agg({
        'is_decreasing': ['sum', 'count', 'mean'],
        'trend_slope': 'mean'
    }).round(4)
    slope_analysis.columns = ['Decreasing_Count', 'Total', 'Success_Rate', 'Avg_Slope']
    print(slope_analysis)
    patterns['slope_difference'] = slope_analysis

    # 4. 복합 패턴 - 가장 효과적인 조합
    print("\n4. BEST COMBINATIONS (Top 10)")
    print("-" * 80)
    decreasing_only = df[df['is_decreasing'] == True].copy()
    if not decreasing_only.empty:
        decreasing_only['combo_desc'] = decreasing_only.apply(
            lambda x: f"Pop{x['pop_ratio']} Age({x['aged_r1']},{x['aged_r2']}) SlopeDiff({x['slope_diff_r1']},{x['slope_diff_r2']})",
            axis=1
        )
        best_combos = decreasing_only.nsmallest(10, 'trend_slope')[
            ['combo_desc', 'trend_slope', 'r_squared', 'mortality_change']
        ]
        print(best_combos.to_string(index=False))
        patterns['best_combinations'] = best_combos

    # 5. 상관관계 분석
    print("\n5. CORRELATION ANALYSIS")
    print("-" * 80)
    numeric_cols = ['pop_r1', 'aged_r1', 'aged_r2', 'slope_diff_r1', 'slope_diff_r2', 
                    'avg_slope_diff', 'trend_slope', 'r_squared']
    correlation = df[numeric_cols].corr()['trend_slope'].sort_values()
    print("Correlation with trend_slope (negative = decreasing mortality):")
    print(correlation)
    patterns['correlation'] = correlation

    return patterns

def save_pattern_analysis(patterns, output_dir):
    """패턴 분석 결과를 파일로 저장"""
    import json
    
    analysis_file = os.path.join(output_dir, 'pattern_analysis.txt')
    with open(analysis_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("PATTERN ANALYSIS RESULTS\n")
        f.write("="*80 + "\n\n")
        
        for pattern_name, pattern_data in patterns.items():
            f.write(f"\n{pattern_name.upper()}\n")
            f.write("-"*80 + "\n")
            if isinstance(pattern_data, pd.DataFrame):
                f.write(pattern_data.to_string())
            else:
                f.write(str(pattern_data))
            f.write("\n\n")
    
    print(f"\nPattern analysis saved to: {analysis_file}")

def main():
    print("="*80)
    print("QUICK SYSTEMATIC EXPERIMENT (OPTIMIZED)")
    print("Finding conditions where mortality DECREASES as epsilon increases")
    print("="*80)
    
    # 속도 최적화를 위해 개별 파일 저장 비활성화
    import allocate_masks as alloc_module
    alloc_module.SAVE_INDIVIDUAL_CSV = SAVE_INDIVIDUAL_FILES
    seird.SAVE_INDIVIDUAL_CSV = SAVE_INDIVIDUAL_FILES
    if RUN_NOPRIORITY_MODEL:
        seird_no.SAVE_INDIVIDUAL_CSV = SAVE_INDIVIDUAL_FILES
    
    print(f"\nOptimizations enabled:")
    print(f"  - Individual file saving: {'ON' if SAVE_INDIVIDUAL_FILES else 'OFF (faster)'}")
    print(f"  - Without-priority model: {'ON' if RUN_NOPRIORITY_MODEL else 'OFF (faster)'}")
    print(f"  - Batch save interval: {BATCH_SAVE_INTERVAL} experiments")
    
    # 백업
    original_config = backup_conditions()
    
    # 실험 조건 생성
    configs = generate_focused_configs()
    configs = define_focused_configs()
    total = len(configs)
    
    print(f"\nTotal experiments: {total}")
    print(f"Estimated time: ~{total*0.7:.0f} minutes (~{total*0.7/60:.1f} hours) with optimizations\n")
    
    # 결과 저장
    all_results = []
    
    for idx, config in enumerate(configs, 1):
        print(f"[{idx}/{total}]", end=" ")
        result = run_single_experiment(config, verbose=True)
        if result:
            all_results.append(result)
        
        # 배치 저장 (더 효율적)
        if idx % BATCH_SAVE_INTERVAL == 0 and all_results:
            df = pd.DataFrame(all_results)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(script_dir, 'output')
            df.to_csv(os.path.join(output_dir, 'quick_experiment_progress.csv'), index=False)
            print(f"\n  >> Progress saved ({len(all_results)} results)")
    
    # 복원
    restore_conditions(original_config)
    
    # 결과 저장 및 분석
    if all_results:
        df = pd.DataFrame(all_results)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'output')
        
        # 전체 결과
        df.to_csv(os.path.join(output_dir, 'quick_experiment_results.csv'), index=False)
        
        # 감소 경향만
        decreasing_df = df[df['is_decreasing'] == True].copy()
        decreasing_df.sort_values('trend_slope', inplace=True)
        decreasing_df.to_csv(os.path.join(output_dir, 'quick_experiment_decreasing.csv'), index=False)
        
        # 패턴 분석 실행 (누적 파일에서 Fair 정책 기반으로 분석)
        patterns = analyze_patterns()
        save_pattern_analysis(patterns, output_dir)
        
        # 요약
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(f"Total experiments: {len(all_results)}")
        print(f"Decreasing trend: {len(decreasing_df)} ({len(decreasing_df)/len(all_results)*100:.1f}%)")
        print(f"Increasing/Flat trend: {len(df) - len(decreasing_df)} ({(len(df) - len(decreasing_df))/len(all_results)*100:.1f}%)")
        
        if not decreasing_df.empty:
            print("\n" + "="*80)
            print("TOP 5 DECREASING TRENDS (Strongest Effect)")
            print("="*80)
            for i, row in decreasing_df.head(5).iterrows():
                rank = list(decreasing_df.index).index(i) + 1
                print(f"\n{rank}. Trend Slope: {row['trend_slope']:.8f} (R²={row['r_squared']:.4f})")
                print(f"   Population: [{row['pop_r1']:,}, {row['pop_r2']:,}] "
                      f"({row['pop_r1']/(row['pop_r1']+row['pop_r2'])*100:.0f}:{row['pop_r2']/(row['pop_r1']+row['pop_r2'])*100:.0f})")
                print(f"   Aged Ratio: R1={row['aged_r1']}, R2={row['aged_r2']}")
                print(f"   Slopes: R1(aged={row['slope_aged_r1']}, non={row['slope_nonaged_r1']}, diff={row['slope_aged_r1']-row['slope_nonaged_r1']}) "
                      f"R2(aged={row['slope_aged_r2']}, non={row['slope_nonaged_r2']}, diff={row['slope_aged_r2']-row['slope_nonaged_r2']})")
                print(f"   Mortality: {row['mortality_start']:.6f} → {row['mortality_end']:.6f} "
                      f"(Δ {row['mortality_change']:.6f})")
        
        print("\n" + "="*80)
        print("FILES SAVED")
        print("="*80)
        print(f"✓ All results: output/quick_experiment_results_1029.csv")
        print(f"✓ Decreasing only: output/quick_experiment_decreasing_1029.csv")
        print(f"✓ Pattern analysis: output/pattern_analysis_1029.txt")
        print(f"✓ Accumulated data: output/policy_results_with_priority_accumulated_1029.csv")
    else:
        print("\nNo successful experiments.")

if __name__ == '__main__':
    main()
