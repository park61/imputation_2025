"""
Simple Interactive widget for exploring configuration space
Population range: 10-100 (in units of 10,000)
Uses on-demand computation
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import (
    interactive, FloatSlider, Dropdown, SelectionSlider,
    VBox, HBox, Label, Output, HTML, Button
)
from IPython.display import display, clear_output
import time

# Add current directory to path
SCRIPT_DIR = Path(__file__).parent if '__file__' in globals() else Path.cwd()
sys.path.append(str(SCRIPT_DIR))

import quick_systematic_experiment as qse
import allocate_masks as alloc
import utils
import seird_model as seird


class SimpleInteractiveExplorer:
    """Simple interactive explorer with population range 10-100 (×10k)"""
    
    def __init__(self):
        self.current_result = None
        self.computation_time = 0
        
        print("🔬 Interactive Configuration Explorer (Simple)")
        print("Population range: 10-100 (×10,000)")
        print("All configurations computed on-demand (~1-5 seconds)")
        print()
    
    def create_widgets(self):
        """Create interactive widgets"""
        
        # Population sliders (10-100 in units of 10,000)
        pop_r1_slider = SelectionSlider(
            options=list(range(10, 101, 10)),
            value=80,
            description='Pop R1 (×10k):',
            continuous_update=False,
            style={'description_width': '130px'},
            layout={'width': '500px'}
        )
        
        pop_r2_slider = SelectionSlider(
            options=list(range(10, 101, 10)),
            value=40,
            description='Pop R2 (×10k):',
            continuous_update=False,
            style={'description_width': '130px'},
            layout={'width': '500px'}
        )
        
        # Aged ratio sliders
        aged_r1_slider = FloatSlider(
            min=0.1, max=0.5, step=0.1, value=0.5,
            description='Aged Ratio R1:',
            continuous_update=False,
            style={'description_width': '130px'},
            layout={'width': '500px'}
        )
        
        aged_r2_slider = FloatSlider(
            min=0.1, max=0.5, step=0.1, value=0.25,
            description='Aged Ratio R2:',
            continuous_update=False,
            style={'description_width': '130px'},
            layout={'width': '500px'}
        )
        
        # Slope dropdowns
        slope_aged_r1 = Dropdown(
            options=[1, 2, 3, 4, 5],
            value=3,
            description='Slope Aged R1:',
            style={'description_width': '130px'},
            layout={'width': '300px'}
        )
        
        slope_nonaged_r1 = Dropdown(
            options=[1, 2, 3, 4, 5],
            value=2,
            description='Slope Non-Aged R1:',
            style={'description_width': '130px'},
            layout={'width': '300px'}
        )
        
        slope_aged_r2 = Dropdown(
            options=[1, 2, 3, 4, 5],
            value=5,
            description='Slope Aged R2:',
            style={'description_width': '130px'},
            layout={'width': '300px'}
        )
        
        slope_nonaged_r2 = Dropdown(
            options=[1, 2, 3, 4, 5],
            value=1,
            description='Slope Non-Aged R2:',
            style={'description_width': '130px'},
            layout={'width': '300px'}
        )
        
        # Output area
        output = Output()
        
        # Status label
        status_label = HTML(value="<b>Status:</b> Ready - Click 'Run Simulation' button to start")
        
        # Run button
        run_button = Button(
            description='▶ Run Simulation',
            button_style='success',
            tooltip='Click to run simulation with current parameters',
            icon='play',
            layout={'width': '200px', 'height': '40px'}
        )
        
        # Store current parameters
        current_params = {
            'pop_r1_unit': 80,
            'pop_r2_unit': 40,
            'aged_r1': 0.5,
            'aged_r2': 0.25,
            'slope_aged_r1': 3,
            'slope_nonaged_r1': 2,
            'slope_aged_r2': 5,
            'slope_nonaged_r2': 1
        }
        
        # Update current parameters when sliders change
        def on_param_change(change):
            param_name = change['owner'].description.split('(')[0].strip().replace(' ', '_').lower()
            if 'pop' in param_name and 'r1' in param_name:
                current_params['pop_r1_unit'] = change['new']
            elif 'pop' in param_name and 'r2' in param_name:
                current_params['pop_r2_unit'] = change['new']
            elif 'aged' in param_name and 'r1' in param_name:
                current_params['aged_r1'] = change['new']
            elif 'aged' in param_name and 'r2' in param_name:
                current_params['aged_r2'] = change['new']
        
        def on_slope_change(change):
            if change['owner'] == slope_aged_r1:
                current_params['slope_aged_r1'] = change['new']
            elif change['owner'] == slope_nonaged_r1:
                current_params['slope_nonaged_r1'] = change['new']
            elif change['owner'] == slope_aged_r2:
                current_params['slope_aged_r2'] = change['new']
            elif change['owner'] == slope_nonaged_r2:
                current_params['slope_nonaged_r2'] = change['new']
        
        # Attach observers
        pop_r1_slider.observe(on_param_change, names='value')
        pop_r2_slider.observe(on_param_change, names='value')
        aged_r1_slider.observe(on_param_change, names='value')
        aged_r2_slider.observe(on_param_change, names='value')
        slope_aged_r1.observe(on_slope_change, names='value')
        slope_nonaged_r1.observe(on_slope_change, names='value')
        slope_aged_r2.observe(on_slope_change, names='value')
        slope_nonaged_r2.observe(on_slope_change, names='value')
        
        # Run button click handler
        def on_run_button_clicked(b):
            pop_r1_unit = current_params['pop_r1_unit']
            pop_r2_unit = current_params['pop_r2_unit']
            aged_r1 = current_params['aged_r1']
            aged_r2 = current_params['aged_r2']
            slope_aged_r1_val = current_params['slope_aged_r1']
            slope_nonaged_r1_val = current_params['slope_nonaged_r1']
            slope_aged_r2_val = current_params['slope_aged_r2']
            slope_nonaged_r2_val = current_params['slope_nonaged_r2']
            
            with output:
                clear_output(wait=True)
                
                # Check slope constraints
                if slope_nonaged_r1_val > slope_aged_r1_val:
                    print("⚠️ Invalid: Slope Non-Aged R1 must be ≤ Slope Aged R1")
                    status_label.value = "<b style='color: red;'>❌ Invalid configuration - check constraints</b>"
                    return
                
                if slope_nonaged_r2_val > slope_aged_r2_val:
                    print("⚠️ Invalid: Slope Non-Aged R2 must be ≤ Slope Aged R2")
                    status_label.value = "<b style='color: red;'>❌ Invalid configuration - check constraints</b>"
                    return
                
                # Convert to actual population (×10,000)
                pop_r1 = pop_r1_unit * 10000
                pop_r2 = pop_r2_unit * 10000
                
                # Status update
                config_str = f"P{pop_r1_unit}×10k_{pop_r2_unit}×10k_A{aged_r1}_{aged_r2}_S{slope_aged_r1_val}_{slope_nonaged_r1_val}_{slope_aged_r2_val}_{slope_nonaged_r2_val}"
                status_html = f"<b>Config:</b> {config_str}<br>"
                status_html += f"<b>Population:</b> R1={pop_r1:,}, R2={pop_r2:,}<br>"
                status_html += f"<b>Status:</b> ⏱️ Computing..."
                status_label.value = status_html
                
                # Compute results
                start_time = time.time()
                result_df = self._compute_config(pop_r1, pop_r2, aged_r1, aged_r2,
                                                 slope_aged_r1_val, slope_nonaged_r1_val, 
                                                 slope_aged_r2_val, slope_nonaged_r2_val)
                self.computation_time = time.time() - start_time
                self.current_result = result_df
                
                # Update status with timing
                status_html = f"<b>Config:</b> {config_str}<br>"
                status_html += f"<b>Population:</b> R1={pop_r1:,}, R2={pop_r2:,}<br>"
                status_html += f"<b style='color: green;'>✅ Computation time:</b> {self.computation_time:.2f} seconds"
                status_label.value = status_html
                
                # Plot
                self._plot_results(result_df)
        
        run_button.on_click(on_run_button_clicked)
        
        # Update function (deprecated - using button now)
        def update_plot(pop_r1_unit, pop_r2_unit, aged_r1, aged_r2, 
                       slope_aged_r1, slope_nonaged_r1, slope_aged_r2, slope_nonaged_r2):
            # This function is kept for compatibility but not used with button approach
            pass
        
        # Layout
        title = HTML(value="<h2>🔬 Interactive Configuration Explorer</h2>")
        
        instructions = HTML(value="""
            <div style='background-color: #e3f2fd; padding: 10px; border-radius: 5px; margin: 10px 0;'>
                <b>📖 Instructions:</b><br>
                1. Adjust sliders to select configuration parameters<br>
                2. <b>Population:</b> 10-100 represents 100,000-1,000,000 (×10,000 units)<br>
                3. Click <b>'▶ Run Simulation'</b> button to compute and display results<br>
                4. <b>Constraints:</b> Non-aged slopes must be ≤ corresponding aged slopes
            </div>
        """)
        
        # Combine all widgets
        ui = VBox([
            title,
            instructions,
            HTML(value="<h3>Configuration Parameters</h3>"),
            HBox([Label("Population (×10k):"), pop_r1_slider]),
            HBox([Label(""), pop_r2_slider]),
            HBox([Label("Aged Ratio:"), aged_r1_slider]),
            HBox([Label(""), aged_r2_slider]),
            HTML(value="<h3>Slope Parameters</h3>"),
            HBox([slope_aged_r1, slope_aged_r2]),
            HBox([slope_nonaged_r1, slope_nonaged_r2]),
            HTML(value="<hr>"),
            HBox([run_button], layout={'justify_content': 'center'}),
            HTML(value="<br>"),
            status_label,
            output
        ])
        
        return ui
    
    def _compute_config(self, pop_r1, pop_r2, aged_r1, aged_r2,
                       slope_aged_r1, slope_nonaged_r1, slope_aged_r2, slope_nonaged_r2):
        """Compute results for a configuration on-demand"""
        
        # Update conditions
        qse.update_conditions(pop_r1, pop_r2, aged_r1, aged_r2,
                              slope_aged_r1, slope_nonaged_r1,
                              slope_aged_r2, slope_nonaged_r2)
        
        # Run pipeline
        allocation_df = alloc.run_allocation_scenarios()
        policy_df = utils.convert_df(allocation_df)
        results_df = seird.run_dynamic_experiment(policy_df)
        
        # Merge quantities
        policy_df_with_qty = policy_df.rename(columns={'region1_masks': 'R1_qty', 'region2_masks': 'R2_qty'})
        policy_df_with_qty['policy_key'] = policy_df_with_qty['policy_name']
        results_df['policy_key'] = results_df.apply(
            lambda row: f"{row['policy']}_{row['supply_rate']}_{row['epsilon']}" if row['policy']=='Fair' and pd.notna(row['epsilon'])
                        else f"{row['policy']}_{row['supply_rate']}",
            axis=1
        )
        results_df = results_df.merge(policy_df_with_qty[['policy_key','R1_qty','R2_qty']], on='policy_key', how='left')
        results_df.drop(columns=['policy_key'], inplace=True)
        
        # Reorder columns
        qty_cols = ['R1_qty', 'R2_qty']
        key_cols = ['epsilon', 'Gini_Value', 'Total_Mortality']
        final_cols = []
        for col in results_df.columns:
            if col in qty_cols or col in key_cols:
                continue
            final_cols.append(col)
            if col == 'policy':
                final_cols.extend([c for c in qty_cols if c in results_df.columns])
            elif col == 'supply_rate':
                final_cols.extend([c for c in key_cols if c in results_df.columns])
        
        return results_df[final_cols]
    
    def _plot_results(self, results_df):
        """Plot results in 3x4 grid (11 Fair plots + 1 comparison), then print summary table"""

        # Fair policy data
        fair_df = results_df[results_df['policy'] == 'Fair'].copy()
        if len(fair_df) == 0:
            print("No Fair policy results found.")
            return

        # Unique supply rates for Fair policy
        supply_rates = sorted(fair_df['supply_rate'].unique())

        # 3x4 grid: first 11 for Fair plots, last (12th) for cross-policy comparison
        n_rows, n_cols = 3, 4
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 15))
        axes = axes.flatten()

        fair_limit = min(11, len(supply_rates))

        # Plot up to 11 Fair plots
        for idx in range(fair_limit):
            rate = supply_rates[idx]
            data = fair_df[fair_df['supply_rate'] == rate]
            axes[idx].scatter(
                data['Gini_Value'], data['Total_Mortality'],
                alpha=0.65, s=90, c='steelblue', edgecolors='black'
            )
            axes[idx].set_xlabel('Gini Value', fontsize=11)
            axes[idx].set_ylabel('Total Mortality', fontsize=11)
            axes[idx].set_title(
                f'Fair Policy: Supply Rate = {rate}\n(n={len(data)} points)',
                fontsize=12, fontweight='bold'
            )
            axes[idx].grid(True, alpha=0.3)

        # Hide any unused Fair slots (positions 0..10), keep position 11 for comparison
        for idx in range(fair_limit, 11):
            axes[idx].set_visible(False)

        # Position 12 (index 11): Cross-Policy Comparison (Equal, Proportional, Utility)
        ax_cross = axes[11]
        policies_to_compare = ['Equal', 'Proportion', 'Utility']
        policy_labels = {
            'Equal': 'Equal',
            'Proportion': 'Proportional',
            'Utility': 'Utility'
        }
        colors = {'Equal': 'blue', 'Proportion': 'green', 'Utility': 'orange'}

        any_plotted = False
        for policy in policies_to_compare:
            policy_df = results_df[results_df['policy'] == policy].copy()
            if len(policy_df) > 0:
                grouped = (
                    policy_df.groupby('supply_rate')['Total_Mortality']
                    .mean()
                    .reset_index()
                    .sort_values('supply_rate')
                )
                ax_cross.plot(
                    grouped['supply_rate'], grouped['Total_Mortality'],
                    marker='o', linewidth=2, markersize=8,
                    label=policy_labels.get(policy, policy),
                    color=colors.get(policy, 'gray')
                )
                any_plotted = True

        ax_cross.set_xlabel('Supply Rate', fontsize=11)
        ax_cross.set_ylabel('Total Mortality', fontsize=11)
        ax_cross.set_title('Cross-Policy Comparison:\nSupply Rate vs Mortality',
                           fontsize=12, fontweight='bold')
        if any_plotted:
            ax_cross.legend(fontsize=10, loc='best')
        ax_cross.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Summary table (print at the end)
        print("="*150)
        print("📊 SUMMARY TABLE - All Policies")
        print("="*150)
        summary_cols = [
            'policy', 'supply_rate', 'epsilon', 'R1_qty', 'R2_qty',
            'R1_Achieved_Utility', 'R2_Achieved_Utility',
            'R1_Total_Utility', 'R2_Total_Utility',
            'Gini_Value', 'Total_Mortality',
            'R1_aged_mortality', 'R1_nonaged_mortality', 'R1_total_mortality',
            'R2_aged_mortality', 'R2_nonaged_mortality', 'R2_total_mortality'
        ]
        available_cols = [col for col in summary_cols if col in results_df.columns]
        summary_df = results_df[available_cols].copy()
        if 'epsilon' in summary_df.columns:
            summary_df = summary_df.sort_values(['policy', 'supply_rate', 'epsilon'])
        else:
            summary_df = summary_df.sort_values(['policy', 'supply_rate'])
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_rows', 100)
        print(summary_df.to_string(index=False))
        print("="*150)
        print()
    
    def display(self):
        """Display the interactive widget"""
        ui = self.create_widgets()
        display(ui)


def create_explorer():
    """Convenience function to create and display explorer"""
    explorer = SimpleInteractiveExplorer()
    return explorer


if __name__ == '__main__':
    print("This module is designed to be used in Jupyter notebooks.")
    print("\nUsage in notebook:")
    print("```python")
    print("import interactive_explorer_simple as ies")
    print("explorer = ies.create_explorer()")
    print("explorer.display()")
    print("```")
