"""
Interactive widget for exploring configuration space
Uses ipywidgets for interactive parameter selection with on-demand computation
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import (
    interactive, FloatSlider, Dropdown, SelectionSlider,
    VBox, HBox, Label, Output, HTML
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


class InteractiveExplorer:
    """Interactive explorer for configuration space with on-demand computation"""
    
    def __init__(self):
        self.current_result = None
        self.computation_time = 0
        
        print("🔬 Interactive Configuration Explorer")
        print("All configurations computed on-demand (~1-5 seconds)")
        print()
    
    def create_widgets(self):
        """Create interactive widgets"""
        
        # Population sliders (in units of 10,000)
        pop_r1_values = list(range(10, 100 + 10, 10))  # 10 to 100 in steps of 10
        pop_r1_slider = SelectionSlider(
            options=pop_r1_values,
            value=80,
            description='Pop R1 (×10k):',
            continuous_update=False,
            style={'description_width': '120px'},
            layout={'width': '500px'}
        )
        
        pop_r2_values = list(range(10, 100 + 10, 10))  # 10 to 100 in steps of 10
        pop_r2_slider = SelectionSlider(
            options=pop_r2_values,
            value=40,
            description='Pop R2 (×10k):',
            continuous_update=False,
            style={'description_width': '120px'},
            layout={'width': '500px'}
        )
        
        # Aged ratio sliders
        aged_r1_slider = FloatSlider(
            min=0.1, max=0.5, step=0.1, value=0.5,
            description='Aged Ratio R1:',
            continuous_update=False,
            style={'description_width': '120px'},
            layout={'width': '500px'}
        )
        
        aged_r2_slider = FloatSlider(
            min=0.1, max=0.5, step=0.1, value=0.25,
            description='Aged Ratio R2:',
            continuous_update=False,
            style={'description_width': '120px'},
            layout={'width': '500px'}
        )
        
        # Slope dropdowns
        slope_aged_r1 = Dropdown(
            options=[1, 2, 3, 4, 5],
            value=3,
            description='Slope Aged R1:',
            style={'description_width': '120px'},
            layout={'width': '300px'}
        )
        
        slope_nonaged_r1 = Dropdown(
            options=[1, 2, 3, 4, 5],
            value=2,
            description='Slope Non-Aged R1:',
            style={'description_width': '120px'},
            layout={'width': '300px'}
        )
        
        slope_aged_r2 = Dropdown(
            options=[1, 2, 3, 4, 5],
            value=5,
            description='Slope Aged R2:',
            style={'description_width': '120px'},
            layout={'width': '300px'}
        )
        
        slope_nonaged_r2 = Dropdown(
            options=[1, 2, 3, 4, 5],
            value=1,
            description='Slope Non-Aged R2:',
            style={'description_width': '120px'},
            layout={'width': '300px'}
        )
        
        # Output area
        output = Output()
        
        # Status label
        status_label = HTML(value="<b>Status:</b> Ready")
        
        # Update function
        def update_plot(pop_r1_unit, pop_r2_unit, aged_r1, aged_r2, 
                       slope_aged_r1, slope_nonaged_r1, slope_aged_r2, slope_nonaged_r2):
            
            with output:
                clear_output(wait=True)
                
                # Check slope constraints
                if slope_nonaged_r1 > slope_aged_r1:
                    print("⚠️ Invalid: Slope Non-Aged R1 must be ≤ Slope Aged R1")
                    return
                
                if slope_nonaged_r2 > slope_aged_r2:
                    print("⚠️ Invalid: Slope Non-Aged R2 must be ≤ Slope Aged R2")
                    return
                
                # Convert to actual population (×10,000)
                pop_r1 = pop_r1_unit * 10000
                pop_r2 = pop_r2_unit * 10000
                
                # Status update
                config_str = f"P{pop_r1_unit}×10k_{pop_r2_unit}×10k_A{aged_r1}_{aged_r2}_S{slope_aged_r1}_{slope_nonaged_r1}_{slope_aged_r2}_{slope_nonaged_r2}"
                status_html = f"<b>Config:</b> {config_str}<br>"
                status_html += f"<b>Population:</b> R1={pop_r1:,}, R2={pop_r2:,}<br>"
                status_html += f"<b>Status:</b> ⏱️ Computing..."
                status_label.value = status_html
                
                # Compute results
                start_time = time.time()
                result_df = self._compute_config(pop_r1, pop_r2, aged_r1, aged_r2,
                                                 slope_aged_r1, slope_nonaged_r1, 
                                                 slope_aged_r2, slope_nonaged_r2)
                self.computation_time = time.time() - start_time
                self.current_result = result_df
                
                # Update status with timing
                status_html = f"<b>Config:</b> {config_str}<br>"
                status_html += f"<b>Population:</b> R1={pop_r1:,}, R2={pop_r2:,}<br>"
                status_html += f"<b>Computation time:</b> {self.computation_time:.2f} seconds"
                status_label.value = status_html
                
                # Plot
                self._plot_results(result_df)
        
        # Create interactive widget
        interactive_plot = interactive(
            update_plot,
            pop_r1_unit=pop_r1_slider,
            pop_r2_unit=pop_r2_slider,
            aged_r1=aged_r1_slider,
            aged_r2=aged_r2_slider,
            slope_aged_r1=slope_aged_r1,
            slope_nonaged_r1=slope_nonaged_r1,
            slope_aged_r2=slope_aged_r2,
            slope_nonaged_r2=slope_nonaged_r2
        )
        
        # Layout
        title = HTML(value="<h2>🔬 Interactive Configuration Explorer</h2>")
        
        instructions = HTML(value="""
            <div style='background-color: #e3f2fd; padding: 10px; border-radius: 5px; margin: 10px 0;'>
                <b>📖 Instructions:</b><br>
                1. Adjust sliders to select configuration parameters<br>
                2. Plots update automatically when you release the slider<br>
                3. Each configuration computes on-demand (~1-5 seconds)<br>
                4. <b>Constraints:</b> Non-aged slopes must be ≤ corresponding aged slopes
            </div>
        """)
        
        # Combine all widgets
        ui = VBox([
            title,
            instructions,
            HTML(value="<h3>Configuration Parameters</h3>"),
            HBox([Label("Population:"), pop_r1_slider]),
            HBox([Label(""), pop_r2_slider]),
            HBox([Label("Aged Ratio:"), aged_r1_slider]),
            HBox([Label(""), aged_r2_slider]),
            HTML(value="<h3>Slope Parameters</h3>"),
            HBox([slope_aged_r1, slope_aged_r2]),
            HBox([slope_nonaged_r1, slope_nonaged_r2]),
            HTML(value="<hr>"),
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
        """Plot results for Fair policy"""
        
        # Filter Fair policy
        fair_df = results_df[results_df['policy'] == 'Fair'].copy()
        
        if len(fair_df) == 0:
            print("No Fair policy results found.")
            return
        
        # Get unique supply rates
        supply_rates = sorted(fair_df['supply_rate'].unique())
        
        # Create subplots
        n_cols = min(3, max(1, len(supply_rates)))
        n_rows = (len(supply_rates) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        
        if len(supply_rates) == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for idx, rate in enumerate(supply_rates):
            data = fair_df[fair_df['supply_rate'] == rate]
            axes[idx].scatter(data['Gini_Value'], data['Total_Mortality'], 
                            alpha=0.65, s=90, c='steelblue', edgecolors='black')
            axes[idx].set_xlabel('Gini Value', fontsize=11)
            axes[idx].set_ylabel('Total Mortality', fontsize=11)
            axes[idx].set_title(f'Supply Rate = {rate}\n(n={len(data)} points)', 
                              fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(supply_rates), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # Show summary statistics
        print("\n" + "="*70)
        print("Summary Statistics (Fair Policy)")
        print("="*70)
        summary = fair_df.groupby('supply_rate').agg({
            'Gini_Value': ['min', 'max', 'mean'],
            'Total_Mortality': ['min', 'max', 'mean'],
            'epsilon': 'count'
        }).round(4)
        print(summary)
    
    def display(self):
        """Display the interactive widget"""
        ui = self.create_widgets()
        display(ui)


def create_interactive_explorer():
    """Convenience function to create and display explorer"""
    explorer = InteractiveExplorer()
    return explorer


if __name__ == '__main__':
    print("This module is designed to be used in Jupyter notebooks.")
    print("\nUsage in notebook:")
    print("```python")
    print("import interactive_explorer as ie")
    print("explorer = ie.create_interactive_explorer()")
    print("explorer.display()")
    print("```")
