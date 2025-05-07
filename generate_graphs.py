import json
from matplotlib.font_manager import font_family_aliases
import matplotlib.pyplot as plt
import numpy as np
import os, sys


class Hypotheses1:
    def __init__(self, data_file, output_dir):
        self.output_dir = output_dir

        self.label_fontsize = 16
        self.title_fontsize = 20
        self.axis_fontsize = 14

        try:
            with open(data_file) as f:
                self.data = json.load(f)
        except IOError as e:
            print(f"Error occurred while reading hypotheses 1 result data: {e}")
            sys.exit(1)

        self.node_nums = self.data["numberOfNodes"]
        self.managed_flood_data = self.data['MANAGED_FLOOD']
        self.gossip_data = self.data["GOSSIP"]
        self.p_values = sorted({entry["p"] for entry in self.gossip_data})
        self.k_values = sorted({entry["k"] for entry in self.gossip_data})


    def plot_all(self):
        self.plot_coverage_by_k()
        self.plot_redundancy_by_k()
        self.plot_std_dev()
        for num in [25, 50, 100, 200]:
            self.plot_redundancy_vs_p(num)

    def plot_coverage_by_k(self):
        results_dir = os.path.join(self.output_dir, "coverage_by_k")
        os.makedirs(results_dir, exist_ok=True)

        for k in self.k_values:
            gossip_k_entries = [entry for entry in self.gossip_data if entry["k"] == k]
            
            # Plot Reachability vs Network Size
            plt.figure(figsize=(8, 6))
            plt.plot(
                self.node_nums, self.managed_flood_data["reachability"], 
                label="Managed Flood", marker="o", linestyle="--"
            )
            for entry in gossip_k_entries:
                plt.plot(
                    self.node_nums, entry["reachability"], 
                    marker="o", label=f"GOSSIP p={entry['p']:.2f}"
                )
            plt.xlabel("Number of Nodes", fontsize=self.label_fontsize)
            plt.ylabel("Coverage (%)", fontsize=self.label_fontsize)
            plt.title(f"Coverage Comparison (k={k})", fontsize=self.title_fontsize)
            plt.xticks(self.node_nums, fontsize=self.axis_fontsize)
            yticks = np.arange(
                0, 
                100,
                10
            )
            plt.yticks(yticks, fontsize=self.axis_fontsize)
            plt.legend(fontsize=self.axis_fontsize)
            plt.grid(True)
            plt.savefig(os.path.join(results_dir, f"coverage_k_{k}.png"))
            plt.close()


    def plot_redundancy_by_k(self):
        results_dir = os.path.join(self.output_dir, "redundancy_by_k")
        os.makedirs(results_dir, exist_ok=True)

        for k in self.k_values:
            gossip_k_entries = [entry for entry in self.gossip_data if entry["k"] == k]

            # Plot Redundancy vs Network Size
            plt.figure(figsize=(8, 6))
            plt.plot(self.node_nums, self.managed_flood_data["redundancy"], 
                     label="Managed Flood", marker="o", linestyle="--")
            for entry in gossip_k_entries:
                plt.plot(self.node_nums, entry["redundancy"], 
                         marker="o", label=f"GOSSIP p={entry['p']:.2f}")
            plt.xlabel("Number of Nodes", fontsize=self.label_fontsize)
            plt.ylabel("Redundancy (%)", fontsize=self.label_fontsize)
            plt.title(f"Redundancy Comparison (k={k})", fontsize=self.title_fontsize)
            plt.xticks(self.node_nums, fontsize=self.axis_fontsize)
            plt.legend(fontsize=self.axis_fontsize)
            plt.grid(True)
            plt.savefig(os.path.join(results_dir, f"redundancy_k_{k}.png"))
            plt.close()

    def plot_coverage_by_p(self):
        results_dir = os.path.join(self.output_dir, "coverage_by_p")
        os.makedirs(results_dir, exist_ok=True)

        for p in self.p_values:
            gossip_p_entries = [entry for entry in self.gossip_data if entry["p"] == p]
            
            # Plot Reachability vs Network Size for different k's
            plt.figure(figsize=(8, 6))
            plt.plot(
                self.node_nums, self.managed_flood_data["reachability"], 
                label="Managed Flood", marker="o", linestyle="--"
            )
            for entry in gossip_p_entries:
                plt.plot(
                    self.node_nums, entry["reachability"], 
                    marker="o", label=f"GOSSIP k={entry['k']}"
                )
            plt.xlabel("Number of Nodes", fontsize=self.label_fontsize)
            plt.ylabel("Reachability (%)", fontsize=self.label_fontsize)
            plt.title(f"Reachability Comparison (p={p})", fontsize=self.title_fontsize)
            plt.xticks(self.node_nums, fontsize=self.axis_fontsize)
            plt.legend(fontsize=self.axis_fontsize)
            plt.grid(True)
            plt.savefig(os.path.join(results_dir, f"coverage_p_{p:.2f}.png"))
            plt.close()

    def plot_redundancy_by_p(self):
        results_dir = os.path.join(self.output_dir, "redundancy_by_p")
        os.makedirs(results_dir, exist_ok=True)

        for p in self.p_values:
            gossip_p_entries = [entry for entry in self.gossip_data if entry["p"] == p]

            # Plot Redundancy vs Network Size for different k's
            plt.figure(figsize=(8, 6))
            plt.plot(self.node_nums, self.managed_flood_data["redundancy"], 
                     label="Managed Flood", marker="o", linestyle="--")
            for entry in gossip_p_entries:
                plt.plot(self.node_nums, entry["redundancy"], 
                         marker="o", label=f"GOSSIP k={entry['k']}")
            plt.xlabel("Number of Nodes", fontsize=self.label_fontsize)
            plt.ylabel("Redundancy (%)", fontsize=self.label_fontsize)
            plt.title(f"Redundancy Comparison (p={p})", fontsize=self.title_fontsize)
            plt.xticks(self.node_nums, fontsize=self.axis_fontsize)
            plt.legend(fontsize=self.axis_fontsize)
            plt.grid(True)
            plt.savefig(os.path.join(results_dir, f"redundancy_p_{p}.png"))
            plt.close()

    def plot_std_dev(self):
        results_dir = os.path.join(self.output_dir, "std_dev")  # Changed directory name
        os.makedirs(results_dir, exist_ok=True)
        
        for k in self.k_values:
            gossip_k_entries = [entry for entry in self.gossip_data if entry["k"] == k]

            # Plot Reachability Standard Deviation
            plt.figure(figsize=(8, 6))
            
            # Directly use std values instead of squaring
            mf_reachability_std = self.managed_flood_data["reachability_stds"]  # No squaring
            plt.plot(
                self.node_nums, mf_reachability_std, 
                label="Managed Flood", marker="o", linestyle="--"
            )

            for entry in gossip_k_entries:
                gossip_reachability_std = entry["reachability_stds"]  # No squaring
                plt.plot(
                    self.node_nums, gossip_reachability_std, 
                    marker="o", label=f"GOSSIP p={entry['p']}"
                )

            plt.xlabel("Number of Nodes", fontsize=self.label_fontsize)
            plt.ylabel("Coverage Std Dev", fontsize=self.label_fontsize)  # Fixed label
            plt.title(f"Coverage Standard Deviation (k={k})", fontsize=self.title_fontsize)  # Fixed title
            plt.xticks(self.node_nums, fontsize=self.axis_fontsize)
            plt.legend(fontsize=self.axis_fontsize)
            plt.grid(True)
            plt.savefig(os.path.join(results_dir, f"reachability_std_k_{k}.png"))  # Fixed filename
            plt.close()

            # Plot Redundancy Standard Deviation
            plt.figure(figsize=(8, 6))
            
            mf_redundancy_std = self.managed_flood_data["redundancy_stds"]  # No squaring
            plt.plot(
                self.node_nums, mf_redundancy_std, 
                label="Managed Flood", marker="o", linestyle="--"
            )
            for entry in gossip_k_entries:
                gossip_redundancy_std = entry["redundancy_stds"]  # No squaring
                plt.plot(
                    self.node_nums, gossip_redundancy_std, 
                    marker="o", label=f"GOSSIP p={entry['p']}"
                )
            plt.xlabel("Number of Nodes", fontsize=self.label_fontsize)
            plt.ylabel("Redundancy Std Dev", fontsize=self.label_fontsize)  # Fixed label
            plt.title(f"Redundancy Standard Deviation (k={k})", fontsize=self.title_fontsize)  # Fixed title
            plt.xticks(self.node_nums, fontsize=self.axis_fontsize)
            plt.legend(fontsize=self.axis_fontsize)
            plt.grid(True)
            plt.savefig(os.path.join(results_dir, f"redundancy_std_k_{k}.png"))  # Fixed filename
            plt.close()

    def plot_redundancy_vs_p(self, node_num):
        import os
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator

        results_dir = os.path.join(self.output_dir, "redundancy_vs_p")
        os.makedirs(results_dir, exist_ok=True)
        node_index = self.node_nums.index(node_num)

        # Create a single combined plot
        fig, ax = plt.subplots(figsize=(10, 6))

        for k in self.k_values:
            # Filter and sort entries for current k
            entries = [entry for entry in self.gossip_data if entry['k'] == k]
            sorted_entries = sorted(entries, key=lambda x: x['p'])

            # Extract redundancy data at 100 nodes
            redundancies = [entry['redundancy'][node_index] for entry in sorted_entries]

            # Plot each k on the same axes
            ax.plot(
                self.p_values,
                redundancies,
                marker='o',
                linestyle='-'
                ,
                linewidth=2,
                label=f'GOSSIP (k={k})'
            )

        # Formatting
        ax.set_title(f"Redundancy vs Probability (p) at {node_num} Nodes", fontsize=self.title_fontsize)
        ax.set_xlabel("p", fontsize=self.label_fontsize)
        ax.set_ylabel("Redundancy (%)", fontsize=self.label_fontsize)
        ax.set_xticks(self.p_values)
        ax.tick_params(axis="both", labelsize=self.axis_fontsize)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=self.axis_fontsize, title="k-values")

        # Save combined figure
        combined_path = os.path.join(results_dir, f'redundancy_vs_p_all_k_{node_num}_nodes.png')
        fig.savefig(combined_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

class Hypotheses2:
    def __init__(self, data_file, output_dir):
        self.output_dir = output_dir

        try:
            with open(data_file) as f:
                self.data = json.load(f)
        except IOError as e:
            print(f"Error occurred while reading hypotheses 2 result data: {e}")
            sys.exit(1)

        self.label_fontsize = 16
        self.title_fontsize = 20
        self.axis_fontsize = 14

        self.managed_flood_data = self.data["MANAGED_FLOOD"]
        self.gossip_data = self.data["GOSSIP"]
        self.p_values = sorted({entry["p"] for entry in self.gossip_data})
        self.k_values = sorted({entry["k"] for entry in self.gossip_data})

        self.movement_ratios = sorted({entry["movement_ratio"] for entry in self.managed_flood_data})
        self.movement_speeds = sorted({entry["movement_speed"] for entry in self.managed_flood_data})


    def plot_all(self):
        self.plot_gossip_vs_managed()
        self.plot_best_variants()
        self.plot_relative_difference()
        self.plot_standard_deviation_comparison()


    def plot_gossip_vs_managed(self):
        output_dir = os.path.join(self.output_dir, "basic_comparison_graphs")
        os.makedirs(output_dir, exist_ok=True)

        for movement_ratio in self.movement_ratios:
            plt.figure(figsize=(8, 6))

            # Filter Managed Flood data for this movement ratio
            mf_data = [
                entry for entry in self.managed_flood_data
                if entry["movement_ratio"] == movement_ratio
            ]

            mf_speeds = [entry["movement_speed"] for entry in mf_data]
            mf_reach = [entry["reachability"] for entry in mf_data]

            # Plot Managed Flood 
            plt.plot(
                mf_speeds,
                mf_reach,
                '-o',
                color='#2c3250',
                markersize=7,
                linewidth=2,
                label="Managed Flood",
            )

            # Now plot each GOSSIP1(p, k) 
            for p in self.p_values:
                for k in self.k_values:
                    gossip_data = [
                        entry for entry in self.gossip_data
                        if entry["p"] == p and entry["k"] == k and entry["movement_ratio"] == movement_ratio
                    ]

                    if not gossip_data:
                        continue

                    gossip_speeds = [entry["movement_speed"] for entry in gossip_data]
                    gossip_reach = [entry["reachability"] for entry in gossip_data]

                    plt.plot(
                        gossip_speeds,
                        gossip_reach,
                        label=f"GOSSIP1(p={p}, k={k})",
                        linestyle='--',
                        marker='s',
                        markersize=6,
                        linewidth=2,
                        alpha=0.8
                    )

            plt.xlabel("Movement Speed (m/min)", fontsize=self.label_fontsize)
            plt.ylabel("Coverage (%)", fontsize=self.label_fontsize)
            plt.title(
                f"Coverage vs Movement Speed\nMobility Ratio = {int(movement_ratio * 100)}%",
                fontsize=self.title_fontsize
            )
            plt.xticks(self.movement_speeds, fontsize=self.axis_fontsize)
            plt.yticks(fontsize=self.axis_fontsize)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend(fontsize=self.label_fontsize-5)
            plt.tight_layout()

            filename = f"comparison_movement_ratio_{int(movement_ratio * 100)}.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=300)
            plt.close()


    def plot_best_variants(self):
        output_dir = os.path.join(self.output_dir, "best_gossip1_graphs")
        os.makedirs(output_dir, exist_ok=True)

        for movement_ratio in self.movement_ratios:
            plt.figure(figsize=(8, 6))

            # Filter Managed Flood data
            mf_data = [
                entry for entry in self.managed_flood_data
                if entry["movement_ratio"] == movement_ratio
            ]

            mf_speeds = [entry["movement_speed"] for entry in mf_data]
            mf_reach = [entry["reachability"] for entry in mf_data]

            # Plot Managed Flood without error bars
            plt.plot(
                mf_speeds,
                mf_reach,
                label="Managed Flood",
                marker='o',  # Added marker for clarity
                color='#2c3250',
                markersize=7,
                linewidth=2
            )

            # Find the best (p,k) for this movement_ratio
            best_avg_reach = -float('inf')
            best_variant = None

            for p in self.p_values:
                for k in self.k_values:
                    gossip_data = [
                        entry for entry in self.gossip_data
                        if entry["p"] == p and entry["k"] == k and entry["movement_ratio"] == movement_ratio
                    ]

                    if not gossip_data:
                        continue

                    avg_reach = sum(entry["reachability"] for entry in gossip_data) / len(gossip_data)

                    if avg_reach > best_avg_reach:
                        best_avg_reach = avg_reach
                        best_variant = (p, k, gossip_data)

            # Plot the best GOSSIP1 variant
            if best_variant:
                p_best, k_best, gossip_data = best_variant
                gossip_speeds = [entry["movement_speed"] for entry in gossip_data]
                gossip_reach = [entry["reachability"] for entry in gossip_data]

                # Plot GOSSIP1 without error bars
                plt.plot(
                    gossip_speeds,
                    gossip_reach,
                    label=f"GOSSIP1 (p={p_best}, k={k_best})",
                    marker='s',  # Added marker for clarity
                    color='#e74c3c',
                    markersize=6,
                    linewidth=2
                )

            plt.xlabel("Movement Speed (m/min)", fontsize=self.label_fontsize)
            plt.ylabel("Coverage (%)", fontsize=self.label_fontsize)
            plt.title(
                f"Coverage vs Movement Speed\nMobility Ratio = {int(movement_ratio * 100)}%",
                fontsize=self.title_fontsize
            )
            plt.xticks(self.movement_speeds, fontsize=self.axis_fontsize)
            plt.yticks(fontsize=self.axis_fontsize)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend(fontsize=self.label_fontsize)
            plt.tight_layout()

            filename = f"best_gossip1_movement_ratio_{int(movement_ratio * 100)}.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=300)
            plt.close()


    def plot_relative_difference(self):
        output_dir = os.path.join(self.output_dir, "relative_difference_graphs")
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(12, 6))
        colours = ['#E6194B', '#3CB44B', '#4363D8']

        for movement_ratio_idx, movement_ratio in enumerate(self.movement_ratios):
            # Filter Managed Flood data
            mf_data = {
                entry["movement_speed"]: entry
                for entry in self.managed_flood_data
                if entry["movement_ratio"] == movement_ratio
            }

            # Find the best (p,k) for this movement_ratio
            best_avg_reach = -float('inf')
            best_variant = None

            for p in self.p_values:
                for k in self.k_values:
                    gossip_data = [
                        entry for entry in self.gossip_data
                        if entry["p"] == p and entry["k"] == k and entry["movement_ratio"] == movement_ratio
                    ]

                    if not gossip_data:
                        continue

                    avg_reach = sum(entry["reachability"] for entry in gossip_data) / len(gossip_data)

                    if avg_reach > best_avg_reach:
                        best_avg_reach = avg_reach
                        best_variant = (p, k, gossip_data)

            if not best_variant:
                continue

            _, _, gossip_data = best_variant

            # Calculate relative differences
            speeds = []
            differences = []

            for entry in gossip_data:
                speed = entry["movement_speed"]
                gossip_reach = entry["reachability"]
                mf_reach = mf_data[speed]["reachability"]

                difference = 100 * (gossip_reach - mf_reach) / mf_reach

                speeds.append(speed)
                differences.append(difference)

            plt.plot(
                speeds,
                differences,
                marker='o',
                label=f"Mobility Ratio {int(movement_ratio * 100)}%",
                linewidth=2,
                markersize=7,
                color=colours[movement_ratio_idx]
            )

        plt.xticks(fontsize=self.axis_fontsize)
        plt.yticks(fontsize=self.axis_fontsize)
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.xlabel("Movement Speed (m/min)", fontsize=self.label_fontsize)
        plt.ylabel("Relative Difference (Coverage %)", fontsize=self.label_fontsize)
        plt.title(
            "Relative Percentage Coverage Difference: Best GOSSIP1 vs Managed Flood",
            fontsize=self.title_fontsize
        )
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(fontsize=self.label_fontsize)
        plt.tight_layout()

        filename = "relative_difference_plot.png"
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()


    def plot_standard_deviation_comparison(self):
        output_dir = os.path.join(self.output_dir, "std_dev_comparison_graphs")
        os.makedirs(output_dir, exist_ok=True)

        for movement_ratio in self.movement_ratios:
            plt.figure(figsize=(8, 6))

            # Filter Managed Flood data
            mf_data = {
                entry["movement_speed"]: entry
                for entry in self.managed_flood_data
                if entry["movement_ratio"] == movement_ratio
            }

            # Find the best (p,k) variant for this movement_ratio
            best_avg_reach = -float('inf')
            best_variant = None

            for p in self.p_values:
                for k in self.k_values:
                    gossip_data = [
                        entry for entry in self.gossip_data
                        if entry["p"] == p and entry["k"] == k and entry["movement_ratio"] == movement_ratio
                    ]

                    if not gossip_data:
                        continue

                    avg_reach = sum(entry["reachability"] for entry in gossip_data) / len(gossip_data)

                    if avg_reach > best_avg_reach:
                        best_avg_reach = avg_reach
                        best_variant = (p, k, gossip_data)

            if not best_variant:
                continue

            best_p, best_k, gossip_data = best_variant

            speeds = sorted(mf_data.keys())

            mf_stds = [mf_data[speed]["reachability_stds"] for speed in speeds]
            gossip_stds = [
                next(entry["reachability_stds"] for entry in gossip_data if entry["movement_speed"] == speed)
                for speed in speeds

            ]
            # Plotting
            plt.plot(
                speeds,
                mf_stds,
                marker='o',
                label="Managed Flood",
                linewidth=2,
                markersize=7,
                color='#2c3250'
            )

            plt.plot(
                speeds,
                gossip_stds,
                marker='s',
                linestyle='--',
                label=f"GOSSIP1(p={best_p},k={best_k})",
                linewidth=2,
                markersize=7,
                color='#e74c3c'
            )

            plt.xlabel("Movement Speed (m/min)", fontsize=self.label_fontsize)
            plt.ylabel("Standard Deviation of Coverage (%)", fontsize=self.label_fontsize)
            plt.title(
                f"Coverage Std. Deviation Comparison\nMobility Ratio = {int(movement_ratio * 100)}%",
                fontsize=self.title_fontsize
            )
            plt.xticks(self.movement_speeds, fontsize=self.axis_fontsize)
            plt.yticks(fontsize=self.axis_fontsize)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend(fontsize=self.label_fontsize)
            plt.tight_layout()

            filename = f"std_dev_comparison_movement_ratio_{int(movement_ratio * 100)}.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=300)
            plt.close()


class Hypotheses3:
    def __init__(self, data_file, output_dir):
        self.output_dir = output_dir

        self.label_fontsize = 16
        self.title_fontsize = 22
        self.axis_fontsize = 14

        try:
            with open(data_file) as f:
                self.data = json.load(f)
        except IOError as e:
            print(f"Error occurred while reading hypotheses 3 result data: {e}")
            sys.exit(1)

        self.node_nums = self.data["numberOfNodes"]
        self.managed_flood_data = self.data['MANAGED_FLOOD']
        self.gossip_data = self.data["GOSSIP"]
        self.p_values = sorted({entry["p"] for entry in self.gossip_data})
        self.k_values = sorted({entry["k"] for entry in self.gossip_data})


    def plot_all(self):
        self.plot_comparisons()
        self.plot_heatmap_pk()
        self.plot_consolidated_by_k()
        self.plot_consolidated_by_p()
        self.plot_relative_difference_by_p()

    def plot_comparisons(self):
        results_dir = os.path.join(self.output_dir, "comparisons")
        os.makedirs(results_dir, exist_ok=True)

        # Plot each GOSSIP configuration against Managed Flooding
        for gossip_entry in self.gossip_data:
            p = gossip_entry["p"]
            k = gossip_entry["k"]
            gossip_collisions = gossip_entry["collisions"]
            gossip_stds = gossip_entry["collisions_stds"]

            plt.figure(figsize=(8, 6))
            
            # Plot Managed Flooding
            plt.errorbar(
                self.node_nums,
                self.managed_flood_data["collisions"],
                yerr=self.managed_flood_data["collisions_stds"],
                label="Managed Flood",
                marker='o',
                linestyle='--',
                capsize=5
            )

            # Plot GOSSIP
            plt.errorbar(
                self.node_nums,
                gossip_collisions,
                yerr=gossip_stds,
                label=f"GOSSIP (p={p}, k={k})",
                marker='s',
                linestyle='-',
                capsize=5
            )

            # Customize plot
            plt.xlabel("Number of Nodes", fontsize=self.label_fontsize)
            plt.ylabel("Collision Rate", fontsize=self.label_fontsize)
            plt.title(
                f"Collision Rates: GOSSIP(p={p}, k={k}) vs Managed Flood",
                fontsize=self.title_fontsize
            )
            plt.legend()
            plt.grid(True)
            plt.xticks(self.node_nums)

            # Save plot
            filename = f"hypothesis3_p{p}_k{k}.png"
            plt.savefig(
                os.path.join(results_dir, filename),
                bbox_inches="tight"
            )
            plt.close()

    
    def plot_consolidated_by_k(self):
        results_dir = os.path.join(self.output_dir, "plot_by_k")
        os.makedirs(results_dir, exist_ok=True)
        
        # Create one plot per k-value, showing all p-variants vs Managed Flooding
        for k in self.k_values:
            plt.figure(figsize=(8, 6))
            
            # Plot Managed Flooding (baseline)
            plt.plot(
                self.node_nums,
                self.managed_flood_data["collisions"],
                label="Managed Flood",
                marker='o',
                linestyle='--',
                color='black',
                linewidth=2
            )

            # Plot all GOSSIP entries for this k with different p
            p_values = sorted({entry["p"] for entry in self.gossip_data})
            colors = plt.cm.viridis_r(np.linspace(0, 1, len(p_values)))  # Color gradient for p

            for p, color in zip(p_values, colors):
                # Extract data for this (k, p)
                entries = [e for e in self.gossip_data if e["k"] == k and e["p"] == p]
                if not entries:
                    continue
                data = entries[0]
                
                plt.plot(
                    self.node_nums,
                    data["collisions"],
                    label=f"GOSSIP (p={p})",
                    marker='s',
                    linestyle='-',
                    color=color,
                    linewidth=1
                )

            # Customize plot
            plt.xlabel("Number of Nodes", fontsize=self.label_fontsize)
            plt.ylabel("Collision Rate", fontsize=self.label_fontsize)
            plt.title(
                f"Collision Rates for k={k}: GOSSIP vs Managed Flood",
                fontsize=self.title_fontsize
            )

            plt.legend(
                fontsize=self.label_fontsize
            ) 

            plt.grid(True)
            plt.xticks(self.node_nums, fontsize=self.axis_fontsize)
            plt.yticks(fontsize=self.axis_fontsize)

            # Save plot
            filename = f"hypothesis3_k{k}.png"
            plt.savefig(
                os.path.join(results_dir, filename),
                bbox_inches="tight"
            )
            plt.close()


    def plot_consolidated_by_p(self):
        results_dir = os.path.join(self.output_dir, "plot_by_p")
        os.makedirs(results_dir, exist_ok=True)
        
        # Create one plot per k-value, showing all p-variants vs Managed Flooding
        for p in self.p_values:
            plt.figure(figsize=(8, 6))
            
            # Plot Managed Flooding (baseline)
            plt.plot(
                self.node_nums,
                self.managed_flood_data["collisions"],
                label="Managed Flood",
                marker='o',
                linestyle='--',
                color='black',
                linewidth=2
            )

            # Plot all GOSSIP entries for this k with different p
            colors = plt.cm.viridis_r(np.linspace(0, 1, len(self.k_values)))  # Color gradient for p

            for k, color in zip(self.k_values, colors):
                # Extract data for this (k, p)
                entries = [e for e in self.gossip_data if e["k"] == k and e["p"] == p]
                if not entries:
                    continue
                data = entries[0]
                
                plt.plot(
                    self.node_nums,
                    data["collisions"],
                    label=f"GOSSIP (k={k})",
                    marker='s',
                    linestyle='-',
                    color=color,
                    linewidth=1
                )

            # Customize plot
            plt.xlabel("Number of Nodes", fontsize=self.label_fontsize)
            plt.ylabel("Collision Rate", fontsize=self.label_fontsize)
            plt.title(
                f"Collision Rates: GOSSIP vs Managed Flood\np={p}",
                fontsize=self.title_fontsize
            )

            plt.legend(fontsize=self.label_fontsize) 

            plt.grid(True)
            plt.xticks(self.node_nums, fontsize=self.axis_fontsize)
            plt.yticks(fontsize=self.axis_fontsize)

            # Save plot
            filename = f"hypothesis3_p{p}.png"
            plt.savefig(
                os.path.join(results_dir, filename),
                bbox_inches="tight"
            )
            plt.close()


    def plot_relative_difference_by_p(self):
        results_dir = os.path.join(self.output_dir, "relative_diff_by_p")
        os.makedirs(results_dir, exist_ok=True)

        mf_collisions = np.array(self.managed_flood_data["collisions"])

        for p in self.p_values:
            plt.figure(figsize=(8, 6))

            # zero‐line for reference
            plt.axhline(0, color='black', linestyle='--', linewidth=1)

            # color map across k-values
            colors = plt.cm.viridis_r(np.linspace(0, 1, len(self.k_values)))

            for k, color in zip(self.k_values, colors):
                # find the gossip entry for this (p, k)
                entry = next((e for e in self.gossip_data if e["p"] == p and e["k"] == k), None)
                if entry is None:
                    continue

                gossip_coll = np.array(entry["collisions"])
                # compute 100*(gossip - MF)/MF
                rel_diff = 100 * (gossip_coll - mf_collisions) / mf_collisions

                plt.plot(
                    self.node_nums,
                    rel_diff,
                    marker='s',
                    linestyle='-',
                    color=color,
                    linewidth=2,
                    markersize=6,
                    label=f"k={k}"
                )

            plt.xlabel("Number of Nodes", fontsize=self.label_fontsize)
            plt.ylabel("Relative Collision Rate Difference (%)", fontsize=self.label_fontsize)
            plt.title(
                f"Relative Collision Rate Difference vs Managed Flood\n(p={p})",
                fontsize=self.title_fontsize
            )
            plt.grid(True, linestyle="--", alpha=0.6)

            plt.xticks(self.node_nums, fontsize=self.axis_fontsize)
            plt.yticks(fontsize=self.axis_fontsize)

            plt.legend(
                title="GOSSIP1 variants",
                fontsize=self.label_fontsize - 1
            )

            plt.tight_layout()
            filename = f"hypothesis3_rel_diff_p{p}.png"
            plt.savefig(os.path.join(results_dir, filename), dpi=300, bbox_inches="tight")
            plt.close()


    def plot_heatmap_pk(self):
        results_dir = os.path.join(self.output_dir, "heatmaps")
        os.makedirs(results_dir, exist_ok=True)

        p_vals = self.p_values
        k_vals = self.k_values
        collision_matrix = np.zeros((len(p_vals), len(k_vals)))

        for node_idx, node_num in enumerate(self.node_nums):
            for i, p in enumerate(p_vals):
                for j, k in enumerate(k_vals):
                    entry = next((e for e in self.gossip_data if e['p'] == p and e['k'] == k), None)
                    collision_matrix[i, j] = entry['collisions'][node_idx] if entry else np.nan

            plt.figure(figsize=(8, 6))
            plt.imshow(collision_matrix, cmap='viridis', origin='lower')
            plt.xticks(np.arange(len(k_vals)), labels=k_vals, fontsize=self.axis_fontsize+3)
            plt.yticks(np.arange(len(p_vals)), labels=p_vals, fontsize=self.axis_fontsize+3)

            cbar = plt.colorbar(label='Collision Rate (%)')
            cbar.ax.yaxis.label.set_fontsize(self.label_fontsize)

            plt.xlabel('k', fontsize=self.label_fontsize+3)
            plt.ylabel('p', fontsize=self.label_fontsize+3)
            plt.title(f'Collision Rate Heatmap\n{node_num} Nodes', fontsize=self.title_fontsize)
            plt.savefig(os.path.join(results_dir, f'heatmap_{node_num}nodes.png'))
            plt.close()


def main():
    hypotheses1_data = "./out/hypotheses_1/data.json"
    output_dir1 = "./out/hypotheses_1/"

    hypotheses2_data = "./out/hypotheses_2/data.json"
    output_dir2 = "./out/hypotheses_2/"

    hypotheses3_data = "./out/hypotheses_3/data.json"
    output_dir3 = "./out/hypotheses_3/"

    h1 = Hypotheses1(hypotheses1_data, output_dir1)
    h2 = Hypotheses2(hypotheses2_data, output_dir2)
    h3 = Hypotheses3(hypotheses3_data, output_dir3)

    h1.plot_all()

if __name__ == "__main__":
    main()
