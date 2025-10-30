import matplotlib.pyplot as plt
import os
import pandas as pd
     
def draw_graphs():
    """확인용 그래프 그리기"""
    # 현재 스크립트의 디렉토리를 기준으로 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    graphs_dir = os.path.join(script_dir, 'graphs')
    os.makedirs(graphs_dir, exist_ok=True) # 그래프 폴더 생성
    
    input_csv_path = os.path.join(output_dir, 'df_results_output.csv')
    if not os.path.exists(input_csv_path):
        print(f"Error: Result file not found at {input_csv_path}")
        return
        
    df_results = pd.read_csv(input_csv_path)

    # Plot 1: Supply Rate vs Gini Value for each allocation type
    plt.figure(figsize=(10, 6))
    # 'Fair' 시나리오는 제외하고 먼저 그림
    for scenario in [s for s in df_results["Scenario"].unique() if not s.startswith('Fair')]:
        subset = df_results[df_results["Scenario"] == scenario]
        plt.plot(subset["supply_rate"], subset["Gini Value"], marker='o', label=scenario)
    plt.title("Supply Rate vs Gini Value by Allocation Type")
    plt.xlabel("Supply Rate")
    plt.ylabel("Gini Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(graphs_dir, 'gini_vs_supply_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Supply Rate vs Total Utility for each allocation type
    plt.figure(figsize=(10, 6))
    for scenario in [s for s in df_results["Scenario"].unique() if not s.startswith('Fair')]:
        subset = df_results[df_results["Scenario"] == scenario]
        plt.plot(subset["supply_rate"], subset["Total Utility"], marker='o', label=scenario)
    plt.title("Supply Rate vs Total Utility by Allocation Type")
    plt.xlabel("Supply Rate")
    plt.ylabel("Total Utility")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(graphs_dir, 'utility_vs_supply_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Graphs saved to {graphs_dir}")

def draw_fair_graphs():
    # 현재 스크립트의 디렉토리를 기준으로 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    graphs_dir = os.path.join(script_dir, 'graphs')

    input_csv_path = os.path.join(output_dir, 'df_results_output.csv')
    if not os.path.exists(input_csv_path):
        print(f"Error: Result file not found at {input_csv_path}")
        return
        
    df_results = pd.read_csv(input_csv_path)
    df_fair = df_results[df_results['Scenario'].str.startswith('Fair')].copy()

    if df_fair.empty:
        print("No fair allocation data to plot.")
        return

    # Plot 1: Gini Value vs. Total Utility (Pareto Frontier)
    plt.figure(figsize=(10, 6))
    for epsilon in sorted(df_fair["Epsilon"].unique()):
        subset = df_fair[df_fair["Epsilon"] == epsilon]
        plt.plot(subset["Gini Value"], subset["Total Utility"], marker='o', linestyle='-', label=f'Epsilon={epsilon}')
    plt.title("Gini Value vs. Total Utility for Fair Allocation")
    plt.xlabel("Gini Value")
    plt.ylabel("Total Utility")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(graphs_dir, 'fair_pareto_frontier.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Fair allocation graphs saved to {graphs_dir}")