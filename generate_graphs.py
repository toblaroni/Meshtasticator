import json
import matplotlib.pyplot as plt
import numpy as np
import os, sys


class Hypotheses2:
    def __init__(self, data_file, output_dir):
        self.output_dir = output_dir

        try:
            with open(data_file) as f:
                self.data = json.load(f)
        except IOError as e:
            print(f"Error occurred while reading hypotheses 2 result data: {e}")
            sys.exit(1)

        self.label_fontsize = 12
        self.title_fontsize = 14

        self.managed_flood_data = self.data["MANAGED_FLOOD"]
        self.gossip_data = self.data["GOSSIP"]
        self.p_values = sorted({entry["p"] for entry in self.gossip_data})
        self.k_values = sorted({entry["k"] for entry in self.gossip_data})

        self.movement_ratios = sorted({entry["movement_ratio"] for entry in self.managed_flood_data})
        self.movement_speeds = sorted({entry["movement_speed"] for entry in self.managed_flood_data})

    def print_all(self):
        self.gossip_vs_mf()

    def gossip_vs_mf(self):
        results_dir = os.path.join(self.output_dir, "gossip_vs_managed")

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        for movement_ratio in self.movement_ratios:
            plt.figure(figsize=(15, 6))

            # Get the Managed Flood data
            mf_reachability = [ config["reachability"] for config in self.managed_flood_data if config["movement_ratio"] == movement_ratio ]

            for config in self.gossip_data:
                p = config["p"]
                k = config["k"]

                # Plot Managed Flood
                plt.plot(
                    self.movement_speeds,
                    mf_reachability,
                    label='MANAGED_FLOOD',
                    linestyle='-',
                    marker='o',
                    color='#2c3250',
                    linewidth=2,
                    markersize=8
                )

                # Plot GOSSIP
                gossip_reach = [
                    config["reachability"] 
                    for config in self.gossip_data 
                        if config["p"] == p and config["k"] == k and 
                        config["movement_ratio"] == movement_ratio 
                ]

                plt.plot(
                    self.movement_speeds,
                    gossip_reach,
                    label=f'GOSSIP1(p={p}, k={k})',
                    linestyle='--',
                    marker='s',
                    color='#e74c3c',
                    linewidth=2,
                    markersize=8
                )

                plt.xlabel("Movement Speeds (m/min)", fontsize=self.label_fontsize)
                plt.ylabel("Coverage (%)", fontsize=self.label_fontsize)
                plt.title(
                    f"Coverage Percentage vs Movement Speed - {int(movement_ratio*100)}% Mobile Nodes (GOSSIP1(p={p}, k={k})",
                    wrap=True
                )
                plt.xticks(self.movement_speeds)
                plt.grid(True, linestyle="--", alpha=0.7)
                plt.legend()
                plt.tight_layout()
                filename = f'gossip_p{p}_k{k}_mr{movement_ratio}.png'
                plt.savefig(os.path.join(results_dir, filename), dpi=300)
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
                label='MANAGED_FLOOD',
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
            plt.title(f'Collision Rate Comparison: MANAGED_FLOOD vs GOSSIP (p={p}, k={k})', fontsize=self.title_fontsize)
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
    hypotheses2_data = "./out/hypotheses_2/data.json"
    output_dir2 = "./out/hypotheses_2/"

    hypotheses3_data = "./out/hypotheses_3/data.json"
    output_dir3 = "./out/hypotheses_3/"

    h2 = Hypotheses2(hypotheses2_data, output_dir2)
    h3 = Hypotheses3(hypotheses3_data, output_dir3)

    h2.print_all()

if __name__ == "__main__":
    main()
