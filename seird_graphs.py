import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from conditions import populations, aged_ratio # populations 변수 import
import os

def load_and_merge_data(aged_priority_file, non_priority_file):
    """
    고령층 우선 배정 데이터와 비우선 배정 데이터를 로드하고 병합
    """
    # 고령층 우선 배정 데이터 로드
    df_priority = pd.read_csv(aged_priority_file)
    df_priority['allocation_type'] = 'Age Priority'
    
    # 비우선 배정 데이터 로드
    df_non_priority = pd.read_csv(non_priority_file)
    df_non_priority['allocation_type'] = 'Non Priority'
    
    # 비우선 배정 데이터에 aged_priority_ratio 컬럼 추가 (균등 배분이므로 인구비율과 동일)
    df_non_priority['aged_priority_ratio'] = np.nan
    
    # 비우선 배정 데이터에 마스크 착용률 컬럼들이 없으면 NaN으로 추가
    mask_rate_columns = ['R1_Aged_MaskRate', 'R1_NonAged_MaskRate', 'R2_Aged_MaskRate', 'R2_NonAged_MaskRate']
    for col in mask_rate_columns:
        if col not in df_non_priority.columns:
            df_non_priority[col] = np.nan
    
    # 두 데이터프레임 병합
    df_combined = pd.concat([df_priority, df_non_priority], ignore_index=True)
    
    return df_combined

def plot_mortality_comparison_by_supply_rate(df, graphs_dir):
    """
    공급률별 사망률 비교 그래프 (정책별 분리, 개별 파일 저장)
    """
    # Extract policy type (Equal, Proportion, Utility)
    df['policy_type'] = df['policy'].apply(lambda x: x.split('_')[0])
    
    # Filter for relevant policies
    policies_to_plot = ['Equal', 'Proportion', 'Utility']
    plot_df = df[df['policy_type'].isin(policies_to_plot)].copy()

    # Mortality columns to plot
    mortality_columns = ['Total_Mortality', 'Region1', 'Region2', 'Region1_Aged', 'Region2_Aged', 'Region1_NonAged', 'Region2_NonAged']
    titles = ['Total Mortality', 'Region 1 Total', 'Region 2 Total', 'Region 1 Aged', 'Region 2 Aged', 'Region 1 Non-Aged', 'Region 2 Non-Aged']
    
    for col, title in zip(mortality_columns, titles):
        plt.figure(figsize=(10, 6))  # 각 플롯에 대해 새로운 Figure 생성
        ax = plt.gca()
        
        # hue와 style을 교체하여 우선순위 배분 여부에 따른 비교를 용이하게 함
        sns.lineplot(data=plot_df, x='supply_rate', y=col, 
                     hue='allocation_type', 
                     style='policy_type', 
                     markers=True, markersize=7,
                     ax=ax, style_order=policies_to_plot)
        
        ax.set_title(f'Mortality Rate by Supply Rate: {title}')
        ax.set_xlabel('Supply Rate')
        ax.set_ylabel('Mortality Rate')
        ax.grid(True, alpha=0.5)
        ax.legend(title='Type')
        
        plt.tight_layout()
        # 파일명에 사용할 수 있도록 제목을 수정
        filename_title = title.replace(' ', '_').replace(':', '')
        filename=os.path.join(graphs_dir, f'mortality_comparison_{filename_title}.png')  
        plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()  # Figure를 닫아 메모리를 해제하고 화면 표시를 방지

def plot_mortality_reduction_rate_comparison(df, graphs_dir):
    """
    최대 사망률 대비 사망률 감소율 비교 그래프 (개별 파일 저장)
    """
    # Extract policy type (Equal, Proportion, Utility)
    df['policy_type'] = df['policy'].apply(lambda x: x.split('_')[0])
    
    # Filter for relevant policies
    policies_to_plot = ['Equal', 'Proportion', 'Utility']
    plot_df = df[df['policy_type'].isin(policies_to_plot)].copy()

    # Mortality columns to plot
    mortality_columns = ['Total_Mortality', 'Region1', 'Region2', 'Region1_Aged', 'Region2_Aged', 'Region1_NonAged', 'Region2_NonAged']
    titles = ['Total Mortality', 'Region 1 Total', 'Region 2 Total', 'Region 1 Aged', 'Region 2 Aged', 'Region 1 Non-Aged', 'Region 2 Non-Aged']
    
    for col, title in zip(mortality_columns, titles):
        # Find the maximum mortality rate for the current metric
        max_mortality = plot_df[col].max()
        
        # Calculate the reduction rate as a percentage, handle division by zero
        reduction_col_name = f'{col}_ReductionRate'
        if max_mortality > 0:
            plot_df[reduction_col_name] = ((max_mortality - plot_df[col]) / max_mortality) * 100
        else:
            plot_df[reduction_col_name] = 0

        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        
        sns.lineplot(data=plot_df, x='supply_rate', y=reduction_col_name, 
                     hue='allocation_type', 
                     style='policy_type', 
                     markers=True, markersize=7,
                     ax=ax, style_order=policies_to_plot)
        
        ax.set_title(f'Mortality Reduction Rate by Supply Rate: {title}')
        ax.set_xlabel('Supply Rate')
        ax.set_ylabel('Mortality Reduction Rate (%)')
        ax.grid(True, alpha=0.5)
        ax.legend(title='Type')
        
        plt.tight_layout()
        filename_title = title.replace(' ', '_').replace(':', '')
        filename=os.path.join(graphs_dir, f'mortality_reduction_rate_{filename_title}.png')
        plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_fair_policy_heatmap(df, graphs_dir):
    """
    Fair 정책에 대한 사망률 감소율 히트맵 생성 (supply_rate vs epsilon)
    """
    # 'policy_type' 컬럼 생성
    df['policy_type'] = df['policy'].apply(lambda x: str(x).split('_')[0])

    # 'Fair' 정책 데이터만 필터링 (대소문자 구분 없이)
    fair_df = df[df['policy_type'].str.lower() == 'fair'].copy()
    
    if fair_df.empty:
        print("No 'Fair' policy data found to plot.")
        return

    # 'epsilon'과 'supply_rate' 컬럼이 숫자인지 확인하고 NaN 값 처리
    fair_df['epsilon'] = pd.to_numeric(fair_df['epsilon'], errors='coerce')
    fair_df['supply_rate'] = pd.to_numeric(fair_df['supply_rate'], errors='coerce')
    fair_df.dropna(subset=['epsilon', 'supply_rate'], inplace=True)

    # 통합 사망률 계산을 위한 인구 데이터
    pop_r1, pop_r2 = populations[0], populations[1]
    aged_ratio_r1, aged_ratio_r2 = aged_ratio[0], aged_ratio[1]
    pop_r1_aged, pop_r2_aged = pop_r1 * aged_ratio_r1, pop_r2 * aged_ratio_r2
    pop_r1_nonaged, pop_r2_nonaged = pop_r1 * (1 - aged_ratio_r1), pop_r2 * (1 - aged_ratio_r2)
    
    for temp_df in [df, fair_df]:
        if 'Region1_Aged' in temp_df.columns and 'Region2_Aged' in temp_df.columns:
            temp_df['Combined_Aged_Mortality'] = (temp_df['Region1_Aged'] * pop_r1_aged + temp_df['Region2_Aged'] * pop_r2_aged) / (pop_r1_aged + pop_r2_aged)
            temp_df['Combined_NonAged_Mortality'] = (temp_df['Region1_NonAged'] * pop_r1_nonaged + temp_df['Region2_NonAged'] * pop_r2_nonaged) / (pop_r1_nonaged + pop_r2_nonaged)

    # 분석할 사망률 컬럼들
    mortality_columns = ['Total_Mortality', 'Region1', 'Region2', 
                         'Region1_Aged', 'Region2_Aged', 'Region1_NonAged', 'Region2_NonAged',
                         'Combined_Aged_Mortality', 'Combined_NonAged_Mortality']
    titles = ['Total Mortality', 'Region 1 Total', 'Region 2 Total', 
              'Region 1 Aged', 'Region 2 Aged', 'Region 1 Non-Aged', 'Region 2 Non-Aged',
              'Combined Aged Mortality', 'Combined NonAged Mortality']

    for col, title in zip(mortality_columns, titles):
        # Find the maximum mortality rate for the current metric across ALL data for a consistent baseline
        max_mortality = df[col].max()
        
        # Calculate the reduction rate as a percentage
        reduction_col_name = f'{col}_ReductionRate'
        if max_mortality > 0:
            fair_df[reduction_col_name] = ((max_mortality - fair_df[col]) / max_mortality) * 100
        else:
            fair_df[reduction_col_name] = 0

        # Age Priority와 Non Priority에 대한 서브플롯 생성
        fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
        fig.suptitle(f'Fair Policy Mortality Reduction Rate Heatmap: {title}', fontsize=16)

        for i, alloc_type in enumerate(['Age Priority', 'Non Priority']):
            ax = axes[i]
            subset_df = fair_df[fair_df['allocation_type'] == alloc_type]
            
            if subset_df.empty:
                ax.set_title(f'{alloc_type}\n(No Data)')
                ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center')
                continue

            try:
                # 중복된 (epsilon, supply_rate) 조합이 있을 경우 평균값 사용
                pivot_df = subset_df.pivot_table(index='supply_rate', columns='epsilon', values=reduction_col_name, aggfunc='mean')
                
                if pivot_df.empty:
                    raise ValueError("Pivot table is empty.")

                # Higher reduction is better, so use 'viridis'. Format as percentage.
                sns.heatmap(pivot_df, ax=ax, cmap='viridis', annot=True, fmt=".2f", linewidths=.5, cbar_kws={'label': 'Reduction Rate (%)'})
                ax.set_title(f'{alloc_type}')
                ax.set_xlabel('Epsilon')
                ax.set_ylabel('Supply Rate' if i == 0 else '')

            except Exception as e:
                ax.set_title(f'{alloc_type}\n(Error)')
                ax.text(0.5, 0.5, f'Error creating heatmap:\n{e}', ha='center', va='center', wrap=True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename_title = title.replace(' ', '_').replace(':', '')
        filename=os.path.join(graphs_dir, f'fair_policy_heatmap_{filename_title}.png')
        plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_fair_policy_surface(df, graphs_dir):
    """
    Fair 정책에 대한 사망률 감소율 3D Surface 그래프 생성 (supply_rate vs epsilon)
    """
    # 'policy_type' 컬럼 생성
    df['policy_type'] = df['policy'].apply(lambda x: str(x).split('_')[0])

    # 'Fair' 정책 데이터만 필터링 (대소문자 구분 없이)
    fair_df = df[df['policy_type'].str.lower() == 'fair'].copy()
    
    if fair_df.empty:
        print("No 'Fair' policy data found to plot for surface.")
        return

    # 'epsilon'과 'supply_rate' 컬럼이 숫자인지 확인하고 NaN 값 처리
    fair_df['epsilon'] = pd.to_numeric(fair_df['epsilon'], errors='coerce')
    fair_df['supply_rate'] = pd.to_numeric(fair_df['supply_rate'], errors='coerce')
    fair_df.dropna(subset=['epsilon', 'supply_rate'], inplace=True)

    # 통합 사망률 계산을 위한 인구 데이터
    # 1. 전체 인구는 conditions.py의 population을 참조
    pop_r1, pop_r2 = populations[0], populations[1]
    aged_ratio_r1, aged_ratio_r2 = aged_ratio[0], aged_ratio[1]
    pop_r1_aged, pop_r2_aged = pop_r1 * aged_ratio_r1, pop_r2 * aged_ratio_r2
    pop_r1_nonaged, pop_r2_nonaged = pop_r1 * (1 - aged_ratio_r1), pop_r2 * (1 - aged_ratio_r2)
    
    for temp_df in [df, fair_df]:
        if 'Region1_Aged' in temp_df.columns and 'Region2_Aged' in temp_df.columns:
            temp_df['Combined_Aged_Mortality'] = (temp_df['Region1_Aged'] * pop_r1_aged + temp_df['Region2_Aged'] * pop_r2_aged) / (pop_r1_aged + pop_r2_aged)
            temp_df['Combined_NonAged_Mortality'] = (temp_df['Region1_NonAged'] * pop_r1_nonaged + temp_df['Region2_NonAged'] * pop_r2_nonaged) / (pop_r1_nonaged + pop_r2_nonaged)

    # 분석할 사망률 컬럼들
    mortality_columns = ['Total_Mortality', 'Region1', 'Region2', 
                         'Region1_Aged', 'Region2_Aged', 'Region1_NonAged', 'Region2_NonAged',
                         'Combined_Aged_Mortality', 'Combined_NonAged_Mortality']
    titles = ['Total Mortality', 'Region 1 Total', 'Region 2 Total', 
              'Region 1 Aged', 'Region 2 Aged', 'Region 1 Non-Aged', 'Region 2 Non-Aged',
              'Combined Aged Mortality', 'Combined NonAged Mortality']

    for col, title in zip(mortality_columns, titles):
        # Find the maximum mortality rate for the current metric across ALL data for a consistent baseline
        max_mortality = df[col].max()
        
        # Calculate the reduction rate as a percentage
        reduction_col_name = f'{col}_ReductionRate'
        if max_mortality > 0:
            fair_df[reduction_col_name] = ((max_mortality - fair_df[col]) / max_mortality) * 100
        else:
            fair_df[reduction_col_name] = 0

        # Age Priority와 Non Priority에 대한 3D 서브플롯 생성
        fig = plt.figure(figsize=(22, 10))
        fig.suptitle(f'Fair Policy Mortality Reduction Rate Surface: {title}', fontsize=16)

        for i, alloc_type in enumerate(['Age Priority', 'Non Priority']):
            ax = fig.add_subplot(1, 2, i + 1, projection='3d')
            subset_df = fair_df[fair_df['allocation_type'] == alloc_type]
            
            if subset_df.empty or len(subset_df['epsilon'].unique()) < 2 or len(subset_df['supply_rate'].unique()) < 2:
                ax.set_title(f'{alloc_type}\n(Not enough data for 3D plot)')
                continue

            try:
                pivot_df = subset_df.pivot_table(index='supply_rate', columns='epsilon', values=reduction_col_name, aggfunc='mean')
                
                X = pivot_df.columns.values
                Y = pivot_df.index.values
                X, Y = np.meshgrid(X, Y)
                Z = pivot_df.values

                surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
                ax.set_title(f'{alloc_type}')
                ax.set_xlabel('Epsilon')
                ax.set_ylabel('Supply Rate')
                ax.set_zlabel('Reduction Rate (%)')
                fig.colorbar(surf, shrink=0.5, aspect=5, ax=ax, pad=0.1)
            except Exception as e:
                ax.set_title(f'{alloc_type}\n(Error: {e})')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename_title = title.replace(' ', '_').replace(':', '')
        filename=os.path.join(graphs_dir, f'fair_policy_surface_{filename_title}.png')
        plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_fair_policy_by_epsilon(df, graphs_dir):
    """
    Fair 정책에 대해 고정된 supply_rate에서 epsilon 변화에 따른 사망률 감소율 라인 그래프 생성
    """
    # 'policy_type' 컬럼 생성
    df['policy_type'] = df['policy'].apply(lambda x: str(x).split('_')[0])

    # 'Fair' 정책 데이터만 필터링
    fair_df = df[df['policy_type'].str.lower() == 'fair'].copy()
    
    if fair_df.empty:
        print("No 'Fair' policy data found to plot for epsilon analysis.")
        return

    # 'epsilon'과 'supply_rate' 컬럼이 숫자인지 확인하고 NaN 값 처리
    fair_df['epsilon'] = pd.to_numeric(fair_df['epsilon'], errors='coerce')
    fair_df['supply_rate'] = pd.to_numeric(fair_df['supply_rate'], errors='coerce')
    fair_df.dropna(subset=['epsilon', 'supply_rate'], inplace=True)

    # 시각화할 supply_rate 선택 (최대 4개)
    unique_supply_rates = sorted(fair_df['supply_rate'].unique())
    if len(unique_supply_rates) > 4:
        # 균등하게 4개 선택
        indices = np.linspace(0, len(unique_supply_rates) - 1, 4, dtype=int)
        selected_supply_rates = [unique_supply_rates[i] for i in indices]
    else:
        selected_supply_rates = unique_supply_rates
    
    plot_df = fair_df[fair_df['supply_rate'].isin(selected_supply_rates)].copy()

    if plot_df.empty:
        print("Not enough data for selected supply rates to plot epsilon analysis.")
        return

    # 통합 사망률 계산을 위한 인구 데이터
    # 1. 전체 인구는 conditions.py의 population을 참조
    pop_r1, pop_r2 = populations[0], populations[1]
    aged_ratio_r1, aged_ratio_r2 = aged_ratio[0], aged_ratio[1]
    pop_r1_aged, pop_r2_aged = pop_r1 * aged_ratio_r1, pop_r2 * aged_ratio_r2
    pop_r1_nonaged, pop_r2_nonaged = pop_r1 * (1 - aged_ratio_r1), pop_r2 * (1 - aged_ratio_r2)

    # 통합 사망률 컬럼을 df와 plot_df에 추가
    for temp_df in [df, plot_df]:
        if 'Region1_Aged' in temp_df.columns and 'Region2_Aged' in temp_df.columns:
            temp_df['Combined_Aged_Mortality'] = (temp_df['Region1_Aged'] * pop_r1_aged + temp_df['Region2_Aged'] * pop_r2_aged) / (pop_r1_aged + pop_r2_aged)
            temp_df['Combined_NonAged_Mortality'] = (temp_df['Region1_NonAged'] * pop_r1_nonaged + temp_df['Region2_NonAged'] * pop_r2_nonaged) / (pop_r1_nonaged + pop_r2_nonaged)

    # 분석할 사망률 컬럼들
    mortality_columns = ['Total_Mortality', 'Region1', 'Region2', 
                         'Region1_Aged', 'Region2_Aged', 'Region1_NonAged', 'Region2_NonAged',
                         'Combined_Aged_Mortality', 'Combined_NonAged_Mortality']
    titles = ['Total Mortality', 'Region 1 Total', 'Region 2 Total', 
              'Region 1 Aged', 'Region 2 Aged', 'Region 1 Non-Aged', 'Region 2 Non-Aged',
              'Combined Aged Mortality', 'Combined NonAged Mortality']

    for col, title in zip(mortality_columns, titles):
        if col not in plot_df.columns:
            continue
            
        # Find the maximum mortality rate for the current metric across ALL data
        max_mortality = df[col].max()
        
        # Calculate the reduction rate as a percentage
        reduction_col_name = f'{col}_ReductionRate'
        if max_mortality > 0:
            plot_df[reduction_col_name] = ((max_mortality - plot_df[col]) / max_mortality) * 100
        else:
            plot_df[reduction_col_name] = 0

        plt.figure(figsize=(12, 7))
        ax = plt.gca()
        
        sns.lineplot(data=plot_df, x='epsilon', y=reduction_col_name, hue='supply_rate', style='allocation_type', palette='viridis', marker='o', ax=ax)
        
        ax.set_title(f'Fair Policy: Reduction Rate vs. Epsilon for {title}')
        ax.set_xlabel('Epsilon')
        ax.set_ylabel('Mortality Reduction Rate (%)')
        ax.grid(True, alpha=0.5)
        ax.legend(title='Supply Rate / Allocation')
        
        plt.tight_layout()
        filename_title = title.replace(' ', '_').replace(':', '')
        filename=os.path.join(graphs_dir, f'fair_policy_epsilon_analysis_{filename_title}.png')
        plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_mortality_reduction_analysis(df, graphs_dir):
    """
    고령층 우선 배정으로 인한 사망률 감소 분석 (개별 파일 저장)
    """
    # 공급률별 데이터 준비
    supply_rates = sorted(df['supply_rate'].unique())
    reduction_data = []
    
    for rate in supply_rates:
        rate_data = df[df['supply_rate'] == rate]
        priority_data = rate_data[rate_data['allocation_type'] == 'Age Priority']
        non_priority_data = rate_data[rate_data['allocation_type'] == 'Non Priority']
        
        if len(priority_data) > 0 and len(non_priority_data) > 0:
            priority_row = priority_data.iloc[0]
            non_priority_row = non_priority_data.iloc[0]
            
            reduction = {
                'supply_rate': rate,
                'total_reduction': non_priority_row['Total_Mortality'] - priority_row['Total_Mortality'],
                'total_reduction_pct': ((non_priority_row['Total_Mortality'] - priority_row['Total_Mortality']) / non_priority_row['Total_Mortality']) * 100,
                'aged_reduction_r1': non_priority_row['Region1_Aged'] - priority_row['Region1_Aged'],
                'aged_reduction_r2': non_priority_row['Region2_Aged'] - priority_row['Region2_Aged'],
                'nonaged_increase_r1': priority_row['Region1_NonAged'] - non_priority_row['Region1_NonAged'],
                'nonaged_increase_r2': priority_row['Region2_NonAged'] - non_priority_row['Region2_NonAged'],
            }
            reduction_data.append(reduction)
    
    reduction_df = pd.DataFrame(reduction_data)
    
    # --- Plot 1: 총 사망률 감소 ---
    plt.figure(figsize=(8, 6))
    ax1 = plt.gca()
    ax1.bar(reduction_df['supply_rate'], reduction_df['total_reduction'], 
            color='green', alpha=0.7)
    ax1.set_title('Total Mortality Reduction (Age Priority vs Non-Priority)')
    ax1.set_xlabel('Supply Rate')
    ax1.set_ylabel('Mortality Reduction')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    filename=os.path.join(graphs_dir, 'reduction_total_mortality.png')
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot 2: 총 사망률 감소 (퍼센트) ---
    plt.figure(figsize=(8, 6))
    ax2 = plt.gca()
    colors_pct = ['red' if x < 0 else 'blue' for x in reduction_df['total_reduction_pct']]
    bars_pct = ax2.bar(reduction_df['supply_rate'], reduction_df['total_reduction_pct'], 
                       color=colors_pct, alpha=0.7)
    ax2.set_title('Total Mortality Change (%)\n(Blue: Age Priority Better, Red: Non-Priority Better)')
    ax2.set_xlabel('Supply Rate')
    ax2.set_ylabel('Change Percentage')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    for bar, val in zip(bars_pct, reduction_df['total_reduction_pct']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.1),
                 f'{val:.2f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
    plt.tight_layout()
    filename=os.path.join(graphs_dir, 'reduction_total_mortality_percent')
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot 3: 고령층 사망률 감소 ---
    plt.figure(figsize=(8, 6))
    ax3 = plt.gca()
    x = np.arange(len(reduction_df))
    width = 0.35
    ax3.bar(x - width/2, reduction_df['aged_reduction_r1'], width, 
            label='Region 1', alpha=0.8, color='darkgreen')
    ax3.bar(x + width/2, reduction_df['aged_reduction_r2'], width, 
            label='Region 2', alpha=0.8, color='lightgreen')
    ax3.set_title('Aged Group Mortality Reduction by Region')
    ax3.set_xlabel('Supply Rate')
    ax3.set_ylabel('Mortality Reduction')
    ax3.set_xticks(x)
    ax3.set_xticklabels(reduction_df['supply_rate'])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    filename=os.path.join(graphs_dir, 'reduction_aged_mortality')
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot 4: 비고령층 사망률 증가 ---
    plt.figure(figsize=(8, 6))
    ax4 = plt.gca()
    ax4.bar(x - width/2, reduction_df['nonaged_increase_r1'], width, 
            label='Region 1', alpha=0.8, color='darkred')
    ax4.bar(x + width/2, reduction_df['nonaged_increase_r2'], width, 
            label='Region 2', alpha=0.8, color='lightcoral')
    ax4.set_title('Non-Aged Group Mortality Increase by Region')
    ax4.set_xlabel('Supply Rate')
    ax4.set_ylabel('Mortality Increase')
    ax4.set_xticks(x)
    ax4.set_xticklabels(reduction_df['supply_rate'])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    filename=os.path.join(graphs_dir, 'increase_nonaged_mortality')
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return reduction_df

def plot_mask_usage_comparison(df, graphs_dir):
    """
    마스크 착용률 비교 (고령층 우선 배정 데이터만 사용, 개별 파일 저장)
    """
    priority_df = df[df['allocation_type'] == 'Age Priority'].copy()
    
    if priority_df.empty:
        print("No age priority data available for mask usage comparison")
        return
    
    supply_rates = sorted(priority_df['supply_rate'].unique())
    
    # 데이터 준비
    aged_rates_r1 = []
    nonaged_rates_r1 = []
    aged_rates_r2 = []
    nonaged_rates_r2 = []
    
    for rate in supply_rates:
        rate_data = priority_df[priority_df['supply_rate'] == rate]
        if len(rate_data) > 0:
            row = rate_data.iloc[0]
            aged_rates_r1.append(row['R1_Aged_MaskRate'])
            nonaged_rates_r1.append(row['R1_NonAged_MaskRate'])
            aged_rates_r2.append(row['R2_Aged_MaskRate'])
            nonaged_rates_r2.append(row['R2_NonAged_MaskRate'])
    
    x = np.arange(len(supply_rates))
    width = 0.35
    
    # --- Plot 1: Region 1 마스크 착용률 ---
    plt.figure(figsize=(8, 6))
    ax1 = plt.gca()
    ax1.bar(x - width/2, aged_rates_r1, width, label='Aged Group', alpha=0.8, color='navy')
    ax1.bar(x + width/2, nonaged_rates_r1, width, label='Non-Aged Group', alpha=0.8, color='lightblue')
    ax1.set_title('Region 1 Mask Usage Rates (Age Priority Allocation)')
    ax1.set_xlabel('Supply Rate')
    ax1.set_ylabel('Mask Usage Rate')
    ax1.set_xticks(x)
    ax1.set_xticklabels(supply_rates)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    filename=os.path.join(graphs_dir, 'mask_usage_comparison_region1')
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Plot 2: Region 2 마스크 착용률 ---
    plt.figure(figsize=(8, 6))
    ax2 = plt.gca()
    ax2.bar(x - width/2, aged_rates_r2, width, label='Aged Group', alpha=0.8, color='darkred')
    ax2.bar(x + width/2, nonaged_rates_r2, width, label='Non-Aged Group', alpha=0.8, color='lightcoral')
    ax2.set_title('Region 2 Mask Usage Rates (Age Priority Allocation)')
    ax2.set_xlabel('Supply Rate')
    ax2.set_ylabel('Mask Usage Rate')
    ax2.set_xticks(x)
    ax2.set_xticklabels(supply_rates)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    filename=os.path.join(graphs_dir, 'mask_usage_comparison_region2')
    plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_table(df, graphs_dir):
    """
    요약 테이블 생성
    """
    summary_data = []
    
    supply_rates = sorted(df['supply_rate'].unique())
    
    for rate in supply_rates:
        rate_data = df[df['supply_rate'] == rate]
        priority_data = rate_data[rate_data['allocation_type'] == 'Age Priority']
        non_priority_data = rate_data[rate_data['allocation_type'] == 'Non Priority']
        
        if len(priority_data) > 0 and len(non_priority_data) > 0:
            priority_row = priority_data.iloc[0]
            non_priority_row = non_priority_data.iloc[0]
            
            summary = {
                'Supply_Rate': rate,
                'Total_Mortality_NonPriority': non_priority_row['Total_Mortality'],
                'Total_Mortality_Priority': priority_row['Total_Mortality'],
                'Mortality_Reduction': non_priority_row['Total_Mortality'] - priority_row['Total_Mortality'],
                'Reduction_Percentage': ((non_priority_row['Total_Mortality'] - priority_row['Total_Mortality']) / non_priority_row['Total_Mortality']) * 100,
                'Aged_R1_NonPriority': non_priority_row['Region1_Aged'],
                'Aged_R1_Priority': priority_row['Region1_Aged'],
                'Aged_R2_NonPriority': non_priority_row['Region2_Aged'],
                'Aged_R2_Priority': priority_row['Region2_Aged'],
            }
            summary_data.append(summary)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'{graphs_dir}/policy_comparison_summary.csv', index=False)
    
    print("Policy Comparison Summary:")
    print(summary_df.round(6))
    
    return summary_df

def plot_policy_comparison_by_supply_rate(df, graphs_dir, supply_rates=None):
    """
    3) Compare mortality rates for all policies at different mask supply rates
    Output: Multiple files (2 supply rates per file)
    """
    # Get all unique supply rates if not specified
    if supply_rates is None:
        supply_rates = sorted(df['supply_rate'].unique())

    # Define the base save path inside the function
    save_path = os.path.join(graphs_dir, 'policy_comparison_by_supply_rate.png')
    
    # Process in chunks of 2 supply rates per file
    graphs_per_file = 2
    num_files = (len(supply_rates) + graphs_per_file - 1) // graphs_per_file
    
    colors = {'Equal': '#e74c3c', 'Proportion': '#3498db', 
              'Utility': '#2ecc71', 'Fair': '#f39c12'}
    
    for file_idx in range(num_files):
        # Get chunk of supply rates for current file
        start_idx = file_idx * graphs_per_file
        end_idx = min((file_idx + 1) * graphs_per_file, len(supply_rates))
        current_rates = supply_rates[start_idx:end_idx]
        
        # Create figure with appropriate grid
        fig, axes = plt.subplots(len(current_rates), 2, 
                                figsize=(16, 4 * len(current_rates)),
                                squeeze=False)
        fig.suptitle('Policy Comparison at Different Supply Rates', 
                    fontsize=16, fontweight='bold')
        
        # Plot each supply rate in current chunk
        for i, supply_rate in enumerate(current_rates):
            supply_data = df[df['supply_rate'] == supply_rate].copy()
            
            # Separate Fair policy (multiple epsilons)
            fair_data = supply_data[supply_data['policy'] == 'Fair']
            other_data = supply_data[supply_data['policy'] != 'Fair']
            
            for j, region in enumerate(['Region1', 'Region2']):
                ax = axes[i, j]
                
                # Collect data for all policies
                aged_rates, nonaged_rates, policy_labels = [], [], []
                
                # Standard policies
                for policy in ['Equal', 'Proportion', 'Utility']:
                    policy_data = other_data[other_data['policy'] == policy]
                    if not policy_data.empty:
                        row = policy_data.iloc[0]
                        aged_rates.append(row[f'{region}_Aged'])
                        nonaged_rates.append(row[f'{region}_NonAged'])
                        policy_labels.append(policy)
                
                # # Fair policies (multiple epsilons)
                # for _, row in fair_data.iterrows():
                #     aged_rates.append(row[f'{region}_Aged'])
                #     nonaged_rates.append(row[f'{region}_NonAged'])
                #     policy_labels.append(f'Fair (ε={row["epsilon"]})')
                
                # Plotting
                x_pos = np.arange(len(policy_labels))
                width = 0.35
                
                bars1 = ax.bar(x_pos - width/2, aged_rates, width, 
                              label='Aged', alpha=0.8, color='darkred')
                bars2 = ax.bar(x_pos + width/2, nonaged_rates, width, 
                              label='Non-Aged', alpha=0.8, color='darkblue')
                
                ax.set_xlabel('Policy')
                ax.set_ylabel('Mortality Rate')
                ax.set_title(f'{region} - Supply Rate: {supply_rate}')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(policy_labels, rotation=45, ha='right')
                ax.legend()
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for bar in bars1 + bars2:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height,
                           f'{height:.4f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        
        # Save current file
        base, ext = save_path.rsplit('.', 1)
        plt.savefig(f'{base}_{file_idx+1}.{ext}', dpi=300, bbox_inches='tight')
        plt.close()

# 메인 실행 함수
def main_comparison_analysis():
    """
    전체 비교 분석 실행
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    graphs_dir = os.path.join(script_dir, 'graphs')
    
    aged_priority_file = os.path.join(output_dir, 'policy_results_with_priority.csv')
    non_priority_file = os.path.join(output_dir, 'policy_results_without_priority.csv')
    # 데이터 로드 및 병합
    df_combined = load_and_merge_data(aged_priority_file, non_priority_file)
    
    # 그래프 생성
    plot_mortality_comparison_by_supply_rate(df_combined, graphs_dir)
    plot_mortality_reduction_rate_comparison(df_combined, graphs_dir)
    plot_fair_policy_heatmap(df_combined, graphs_dir)
    plot_fair_policy_surface(df_combined, graphs_dir)
    plot_fair_policy_by_epsilon(df_combined, graphs_dir)
    plot_mask_usage_comparison(df_combined, graphs_dir)
    reduction_df = plot_mortality_reduction_analysis(df_combined, graphs_dir)
    plot_policy_comparison_by_supply_rate(df_combined, graphs_dir)
    # 요약 테이블 생성
    #summary_df = create_summary_table(df_combined, graphs_dir)
    
    #return df_combined, reduction_df, summary_df
