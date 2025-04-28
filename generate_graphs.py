import json
from matplotlib.font_manager import font_family_aliases
import matplotlib.pyplot as plt
import numpy as np
import os, sys


class Hypotheses1:
    def __init__(self, data_file, output_dir):
        self.output_dir = output_dir

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
        self.plot_reachability_comparison()
        self.plot_redundancy_comparison()


    def plot_reachability_comparison(self):
        # Create individual plots for each k value
        for k in self.k_values:
            plt.figure(figsize=(8, 6))
            
            # Plot Managed Flooding
            plt.plot(
                self.node_nums,
                self.managed_flood_data['reachability'],
                '-o',
                color='black',
                linewidth=2,
                markersize=8,
                label='Managed Flooding'
            )

            # Plot GOSSIP configurations for this k
            for entry in self.gossip_data:
                if entry['k'] == k:
                    p = entry['p']
                    plt.plot(
                        self.node_nums,
                        entry['reachability'],
                        '--o',
                        alpha=0.8,
                        markersize=6,
                        label=f'GOSSIP p={p}'
                    )

            # Configure plot
            plt.xscale('log')
            plt.xticks(self.node_nums)
            plt.gca().get_xaxis().set_major_formatter(plt.ScalarFormatter())
            plt.xlabel('Number of Nodes (log scale)')
            plt.ylabel('Reachability (%)')
            plt.title(f'Reachability Comparison (k={k})')
            plt.legend()
            plt.grid(True, which='both', linestyle='--')
            
            # Save and close
            plt.savefig(os.path.join(self.output_dir, f'hypothesis1_reachability_k{k}.png'), bbox_inches='tight')
            plt.close()


    def plot_redundancy_comparison(self):

        # Create individual plots for each k value
        for k in self.k_values:
            plt.figure(figsize=(8, 6))
            
            # Plot Managed Flooding
            plt.plot(
                self.node_nums,
                self.managed_flood_data['redundancy'],
                '-o',
                color='black',
                linewidth=2,
                markersize=8,
                label='Managed Flooding'
            )

            # Plot GOSSIP configurations for this k
            for entry in self.gossip_data:
                if entry['k'] == k:
                    p = entry['p']
                    plt.plot(
                        self.node_nums,
                        entry['redundancy'],
                        '--o',
                        alpha=0.8,
                        markersize=6,
                        label=f'GOSSIP p={p}'
                    )

            # Configure plot
            plt.xscale('log')
            plt.xticks(self.node_nums)
            plt.gca().get_xaxis().set_major_formatter(plt.ScalarFormatter())
            plt.xlabel('Number of Nodes (log scale)')
            plt.ylabel('Redundancy (%)')
            plt.title(f'Redundancy Comparison (k={k})')
            plt.legend()
            plt.grid(True, which='both', linestyle='--')
            
            # Save and close
            plt.savefig(os.path.join(self.output_dir, f'hypothesis1_redundancy_k{k}.png'), bbox_inches='tight')
            plt.close()

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
            "Relative Percentage Coverage Difference: Best GOSSIP1 vs Managed Flooding",
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
            gossip_variance = [ stds**2 for stds in gossip_stds ] 
            mf_variance = [ stds**2 for stds in mf_stds ] 

            # Plotting
            plt.plot(
                speeds,
                mf_variance,
                marker='o',
                label="Managed Flood",
                linewidth=2,
                markersize=7,
                color='#2c3250'
            )

            plt.plot(
                speeds,
                gossip_variance,
                marker='s',
                linestyle='--',
                label=f"GOSSIP1(p={best_p},k={best_k})",
                linewidth=2,
                markersize=7,
                color='#e74c3c'
            )

            plt.xlabel("Movement Speed (m/min)", fontsize=self.label_fontsize)
            plt.ylabel("Variance of Coverage (%)", fontsize=self.label_fontsize)
            plt.title(
                f"Coverage Variance Comparison\nMobility Ratio = {int(movement_ratio * 100)}%",
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

        self.label_fontsize = 12
        self.title_fontsize = 14

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
        self.plot_gossip_vs_managed()
        self.plot_collisions_vs_p()
        self.plot_collisions_vs_k()
        self.plot_heatmap_pk()
    
    def plot_gossip_vs_managed(self):
        results_dir = os.path.join(self.output_dir, "gossip_vs_managed")

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        for config in self.gossip_data:
            p = config['p']
            k = config['k']

            plt.figure(figsize=(12, 6))

            plt.plot(
                self.node_nums,
                self.managed_flood_data['collisions'],
                label='Managed Flood',
                linestyle='-',
                marker='o',
                color='#2c3e50',
                linewidth=2,
                markersize=8
            )

            plt.plot(
                self.node_nums,
                config['collisions'],
                label=f'GOSSIP (p={p}, k={k})',
                linestyle='--',
                marker='s',
                color='#e74c3c',
                linewidth=2,
                markersize=8
            )

            plt.xlabel('Number of Nodes', fontsize=self.label_fontsize)
            plt.ylabel('Collision Rate (%)', fontsize=self.label_fontsize)
            plt.title(f'Collision Rate Comparison: Managed Flood vs GOSSIP (p={p}, k={k})', fontsize=self.title_fontsize)
            plt.xticks(self.node_nums, labels=self.node_nums)  # Explicit x-ticks
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()

            # Highlight smaller networks
            ax = plt.gca()
            for x in [3, 5, 10]:
                ax.axvline(x=x, color='#f39c12', linestyle=':', alpha=0.4, linewidth=3)

            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f'gossip_p{p}_k{k}.png'), dpi=300)
            plt.close()
        
    
    def plot_collisions_vs_p(self):
        results_dir = os.path.join(self.output_dir, "collisions_vs_p")

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        for k in self.k_values:
            for node_idx, num_nodes in enumerate(self.node_nums):

                collisions_per_p = []

                for p_val in self.p_values:
                    match = next((entry for entry in self.gossip_data if entry["p"] == p_val and entry["k"] == k), None)
                    if match:
                        collisions_per_p.append(match["collisions"][node_idx])
                    else:
                        collisions_per_p.append(None)

                plt.figure(figsize=(7, 5))
                plt.title(f"Collison Rate vs P Value (k={k}, {num_nodes} nodes)", fontsize=self.title_fontsize)
                plt.xlabel("P Value", fontsize=self.label_fontsize)
                plt.ylabel("Collision Rate (%)", fontsize=self.label_fontsize)
                plt.grid(True, linestyle='--', alpha=0.7)

                plt.plot(self.p_values, collisions_per_p)
                plt.xticks(self.p_values)
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f"collisions_vs_p_k{k}_{num_nodes}nodes.png"))
                plt.close()


    def plot_collisions_vs_k(self):
        results_dir = os.path.join(self.output_dir, "collisions_vs_k")

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Group by p value and compare the effects of k
        for p_val in self.p_values:
            for node_idx, node_num in enumerate(self.node_nums):
                collisions_per_k = []

                for k in self.k_values:
                    match = next((entry for entry in self.gossip_data if entry["p"] == p_val and entry["k"] == k), None)
                    collisions_per_k.append(match["collisions"][node_idx])

                plt.figure(figsize=(7, 5))
                plt.title(f"Collison Rate vs K Value (p={p_val}, {node_num} nodes)", fontsize=self.title_fontsize)
                plt.xlabel("K Value", fontsize=self.label_fontsize)
                plt.ylabel("Collision Rate (%)", fontsize=self.label_fontsize)
                plt.grid(True, linestyle='--', alpha=0.7)

                plt.plot(self.k_values, collisions_per_k)
                plt.xticks(self.k_values)
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f"collisions_vs_k_p{p_val}_{node_num}nodes.png"))
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
            plt.xticks(np.arange(len(k_vals)), labels=k_vals)
            plt.yticks(np.arange(len(p_vals)), labels=p_vals)
            plt.colorbar(label='Collision Rate (%)')
            plt.xlabel('k')
            plt.ylabel('p')
            plt.title(f'Collision Rate Heatmap (Nodes={node_num})')
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
    h2.plot_all()

if __name__ == "__main__":
    main()
