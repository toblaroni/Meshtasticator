#!/usr/bin/env python3
import collections
import time
import matplotlib
import sys, os, json

try:
    matplotlib.use("TkAgg")
except ImportError:
    print('Tkinter is needed. Install python3-tk with your package manager.')
    exit(1)
import simpy
import numpy as np
import random
import matplotlib.pyplot as plt

from lib.hypotheses_1_config import Config1
from lib.common import *
from lib.packet import *
from lib.mac import *
from lib.discrete_event import *
from lib.node import *
from lib.batch_common import *

# Debug
conf = Config1()
VERBOSE = False
SHOW_GRAPH = False
SAVE = True

if VERBOSE:
    def verboseprint(*args, **kwargs): 
        print(*args, **kwargs)
else:
    def verboseprint(*args, **kwargs): 
        pass

#######################################
####### INDEPENDENT VARIABLES ########
#######################################

repetitions = 20
numberOfNodes = [ 5, 25, 50, 100, 200 ]
gossip_p_vals = [ 0.55, 0.6, 0.65, 0.7, 0.75 ]
gossip_k_vals = [ 1, 2, 4 ]

routerTypes = [ (conf.ROUTER_TYPE.MANAGED_FLOOD, None, None) ]
for p in gossip_p_vals:
    for k in gossip_k_vals:
        routerTypes.append( (conf.ROUTER_TYPE.GOSSIP, p, k) )

output_dir = f"./out/hypotheses_1/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# We will collect the metrics in dictionaries keyed by router type.
# For example: collisions_dict[ routerType ] = [list of mean collisions, one per nrNodes]
reachability_dict = {}
reachabilityStds_dict = {}
redundancy_dict = {}
redundancyStds_dict = {}

# Pre-Generate random positions
positions_cache = gen_random_positions(conf, repetitions, numberOfNodes)

# Initialize dictionaries for each router type, individual dictionaries for 
for rt in routerTypes:
    reachability_dict[rt] = []
    reachabilityStds_dict[rt] = []
    redundancy_dict[rt] = []
    redundancyStds_dict[rt] = []

###########################################################
# Main simulation loops
###########################################################

# Outer loop for each router type
for rt_i, routerType in enumerate(routerTypes):
    routerTypeLabel, gossip_p, gossip_k = routerType

    # Prepare arrays for the final plot data, one per metric
    reachability = []
    reachabilityStds = []
    redundancy = []
    redundancyStds = []

    # Inner loop for each nrNodes
    for p, nrNodes in enumerate(numberOfNodes):
        nodeReach = [ 0 for _ in range(repetitions)]
        nodeRedundancy = [ 0 for _ in range(repetitions)]

        if routerTypeLabel == conf.ROUTER_TYPE.MANAGED_FLOOD:
            print(f"\n[Router: {routerTypeLabel}] Start of {p+1} out of {len(numberOfNodes)} - {nrNodes} nodes")
        else:
            print(f"\n[Router: {routerTypeLabel}({gossip_p}, {gossip_k})] Start of {p+1} out of {len(numberOfNodes)} - {nrNodes} nodes")

        for rep in range(repetitions):
            # For the highest degree of separation between runs, config
            # should be instantiated every repetition for this router type and node number
            routerTypeConf = Config1()
            routerTypeConf.SELECTED_ROUTER_TYPE = routerTypeLabel
            routerTypeConf.NR_NODES = nrNodes

            routerTypeConf.GOSSIP_P = gossip_p
            routerTypeConf.GOSSIP_K = gossip_k

            routerTypeConf.updateRouterDependencies()

            effectiveSeed = rt_i * 10000 + rep
            routerTypeConf.SEED = effectiveSeed
            random.seed(effectiveSeed)
            env = simpy.Environment()
            bc_pipe = BroadcastPipe(env)

            # Start the progress-logging process
            env.process(simulationProgress(env, conf, rep, repetitions, routerTypeConf.SIMTIME))

            # Retrieve the pre-generated positions for this (nrNodes, rep)
            coords = positions_cache[(nrNodes, rep)]

            nodes = []
            messages = []
            packets = []
            delays = []
            packetsAtN = [[] for _ in range(routerTypeConf.NR_NODES)]
            messageSeq = {"val": 0}

            if SHOW_GRAPH:
                graph = Graph(routerTypeConf)
            for nodeId in range(routerTypeConf.NR_NODES):
                x, y = coords[nodeId]

                # We create a nodeConfig dict so that MeshNode will use that
                nodeConfig = {
                    'x': x,
                    'y': y,
                    'z': routerTypeConf.HM,
                    'isRouter': False,
                    'isRepeater': False,
                    'isClientMute': False,
                    'hopLimit': routerTypeConf.hopLimit,
                    'antennaGain': routerTypeConf.GL
                }

                node = MeshNode(
                    routerTypeConf, nodes, env, bc_pipe, nodeId, routerTypeConf.PERIOD,
                    messages, packetsAtN, packets, delays, nodeConfig,
                    messageSeq, verboseprint, rep_seed=rep
                )
                nodes.append(node)
                if SHOW_GRAPH:
                    graph.addNode(node)

            if routerTypeConf.MOVEMENT_ENABLED and SHOW_GRAPH:
                env.process(runGraphUpdates(env, graph, nodes))

            totalPairs, symmetricLinks, asymmetricLinks, noLinks = setupAsymmetricLinks(routerTypeConf, nodes)

            # Start simulation
            env.run(until=routerTypeConf.SIMTIME)

            # Calculate stats
            nrSensed = sum([1 for pkt in packets for n in nodes if pkt.sensedByN[n.nodeid]])
            nrReceived = sum([1 for pkt in packets for n in nodes if pkt.receivedAtN[n.nodeid]])
            nrUseful = sum([n.usefulPackets for n in nodes])

            if messageSeq["val"] != 0:
                nodeReach[rep] = nrUseful / (messageSeq["val"] * (routerTypeConf.NR_NODES - 1)) * 100
            else:
                nodeReach[rep] = np.NaN

            if nrReceived > 0:
                nodeRedundancy[rep] = ( ( nrReceived - nrUseful ) / nrReceived ) * 100
            else:
                nodeRedundancy[rep] = np.NaN

        # After finishing all repetitions for this nrNodes, compute means/stdevs
        reachability.append(np.nanmean(nodeReach))
        reachabilityStds.append(np.nanstd(nodeReach))
        redundancy.append(np.nanmean(nodeRedundancy))
        redundancyStds.append(np.nanstd(nodeRedundancy))

    # After finishing all nrNodes for the *current* router type,
    # store these lists in the dictionary so we can plot after.
    reachability_dict[routerType] = reachability
    reachabilityStds_dict[routerType] = reachabilityStds
    redundancy_dict[routerType] = redundancy
    redundancyStds_dict[routerType] = redundancyStds

# Save to json file for easy re-graphing
results = {
    "reachability": reachability_dict,
    "reachability_stds": reachabilityStds_dict,
    "redundancy": redundancy_dict,
    "redundancy_stds": redundancyStds_dict,
}

results_data = {
    "numberOfNodes": numberOfNodes,
    "MANAGED_FLOOD": {},
    "GOSSIP": []    # Explicitly save p and k for easy retrieval
}


for rt_info in routerTypes:
    router_type = rt_info[0]

    if router_type == conf.ROUTER_TYPE.MANAGED_FLOOD:
        for result_key in results:
            results_data["MANAGED_FLOOD"][result_key] = results[result_key][rt_info]

    elif router_type == conf.ROUTER_TYPE.GOSSIP:
        _, p, k = rt_info

        entry = { "p": p, "k": k }

        for result_key in results:
            entry[result_key] = results[result_key][rt_info]

        results_data["GOSSIP"].append(entry)


with open(os.path.join(output_dir, "data.json"), "w") as f:
    json.dump(results_data, f, indent=4)


###########################################################
# Plotting
###########################################################

def router_type_label(rt_info):
    rt, p, k = rt_info
    if rt == conf.ROUTER_TYPE.MANAGED_FLOOD:
        return "Managed Flood"
    elif rt == conf.ROUTER_TYPE.GOSSIP:
        return f"GOSSIP({p}, {k})"
    else:
        return str(rt)

###########################################################
# Choose a baseline router type for comparison
###########################################################
baselineRt = (conf.ROUTER_TYPE.MANAGED_FLOOD, None, None)

# Define colors for p values
colors = plt.cm.viridis(np.linspace(0, 1, len(gossip_p_vals)))

# Plot Coverage and Redundancy
for metric_name, metric_dict, std_dict in [
    ('Coverage', reachability_dict, reachabilityStds_dict),
    ('Redundancy', redundancy_dict, redundancyStds_dict)
]:
    fig, axes = plt.subplots(1, len(gossip_k_vals), figsize=(15, 5), sharey=True)
    fig.suptitle(f'{metric_name} by Node Count and Gossip Parameters', y=1.05)

    # Plot baseline (Managed Flood)
    baseline_vals = metric_dict[baselineRt]
    for ax in axes:
        ax.plot(numberOfNodes, baseline_vals, '--', color='gray', label='Managed Flood')

    # Plot GOSSIP results for each k
    for ax_idx, k in enumerate(gossip_k_vals):
        ax = axes[ax_idx]
        for p_idx, p in enumerate(gossip_p_vals):
            rt_info = (conf.ROUTER_TYPE.GOSSIP, p, k)
            if rt_info not in metric_dict:
                continue

            # Extract values and errors
            vals = metric_dict[rt_info]
            stds = std_dict[rt_info]

            ax.errorbar(
                numberOfNodes, vals, yerr=stds,
                color=colors[p_idx], marker='o', linestyle='-',
                label=f'p={p}'
            )

        ax.set_title(f'k = {k}')
        ax.set_xlabel('Number of Nodes')
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(f'{metric_name} (%)')

    # Combine legends
    handles, labels = axes[0].get_legend_handles_labels()

    plt.tight_layout()

    # Add legend outside all subplots
    fig.legend(
        handles, labels,
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        frameon=False
    )


    plt.savefig(os.path.join(output_dir, f"{metric_name.lower()}_trends.png"), bbox_inches='tight')
    plt.close()
