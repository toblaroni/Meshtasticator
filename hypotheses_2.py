#!/usr/bin/env python3
import collections
import enum
import time
import matplotlib
import sys, os, json

try:
    matplotlib.use("TkAgg")
except ImportError:
    print('Tkinter is needed. Install python3-tk with your package manager.')
    exit(1)
from numpy.lib import nanstd
import simpy
import numpy as np
import random
import matplotlib.pyplot as plt

from lib.hypotheses_1_config import Config
from lib.common import *
from lib.packet import *
from lib.mac import *
from lib.discrete_event import *
from lib.node import *
from lib.batch_common import *

# Debug
conf = Config()
VERBOSE = False
SHOW_GRAPH = False

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
numberOfNodes = [ 30 ]
mobility_ratios = [ 0.8, 0.9, 1 ]
movement_speeds = [ conf.WALKING_METERS_PER_MIN, conf.BIKING_METERS_PER_MIN, conf.DRIVING_METERS_PER_MIN ]
gossip_p_vals = [ 0.55, 0.6, 0.65, 0.7, 0.75 ]
gossip_k_vals = [ 1, 2 ]


routerTypes = [ (conf.ROUTER_TYPE.MANAGED_FLOOD, None, None) ]
for p in gossip_p_vals:
    for k in gossip_k_vals:
        routerTypes.append( (conf.ROUTER_TYPE.GOSSIP, p, k) )

output_dir = f"./out/hypotheses_2/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# We will collect the metrics in dictionaries keyed by router type.
# For example: collisions_dict[ routerType ] = [list of mean collisions, one per nrNodes]
reachability_dict = {}
reachabilityStds_dict = {}

# Pre-Generate random positions
positions_cache = gen_random_positions(conf, repetitions, numberOfNodes)

# Initialize dictionaries for each router type, individual dictionaries for 
for rt in routerTypes:
    reachability_dict[rt] = {
        (mobility_ratio, movement_speed): []
        for mobility_ratio in mobility_ratios
        for movement_speed in movement_speeds
    }

    reachabilityStds_dict[rt] = {
        (mobility_ratio, movement_speed): []
        for mobility_ratio in mobility_ratios
        for movement_speed in movement_speeds
    }

###########################################################
# Main simulation loops
###########################################################

# Outer loop for each router type
for rt_i, routerType in enumerate(routerTypes):
    routerTypeLabel, gossip_p, gossip_k = routerType

    # Prepare arrays for the final plot data, one per metric
    reachability = []
    reachabilityStds = []

    # Inner loop for each nrNodes
    for r, mobility_ratio in enumerate(mobility_ratios):
        for s, movement_speed in enumerate(movement_speeds):

            nodeReach = [ 0 for _ in range(repetitions)]

            for rep in range(repetitions):
                if routerTypeLabel == conf.ROUTER_TYPE.MANAGED_FLOOD:
                    print(f"\n[Router: {routerTypeLabel}] Start of mobility ratio: {mobility_ratio}, movement speed: {movement_speed}m/min.")
                else:
                    print(f"\n[Router: {routerTypeLabel}({gossip_p}, {gossip_k})] Start of mobility ratio: {mobility_ratio}, movement speed: {movement_speed}m/min.")

                # For the highest degree of separation between runs, config
                # should be instantiated every repetition for this router type and node number
                routerTypeConf = Config()
                routerTypeConf.SELECTED_ROUTER_TYPE = routerTypeLabel
                routerTypeConf.NR_NODES = numberOfNodes[0]

                routerTypeConf.APPROX_RATIO_NODES_MOVING = mobility_ratio
                routerTypeConf.MOVEMENT_SPEED = movement_speed

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
                coords = positions_cache[(numberOfNodes[0], rep)]

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

            # After finishing all nrNodes for the *current* router type,
            # store these lists in the dictionary so we can plot after.
            key = (mobility_ratio, movement_speed)
            reachability_dict[routerType][key].append(np.nanmean(nodeReach))
            reachabilityStds_dict[routerType][key].append(np.nanstd(nodeReach))


# Save to json file for easy re-graphing
results = {
    "reachability": reachability_dict,
    "reachability_stds": reachabilityStds_dict,
}

results_data = {
    "MANAGED_FLOOD": [],
    "GOSSIP": [] 
}

for rt_info in routerTypes:
    router_type, gossip_p, gossip_k = rt_info

    if router_type == conf.ROUTER_TYPE.MANAGED_FLOOD:
        for mr in mobility_ratios:
            for ms in movement_speeds:
                entry = {
                    "movement_ratio": mr,
                    "movement_speed": ms,
                }
                for result_key in results:
                    entry[result_key] = results[result_key][rt_info][(mr, ms)][0]

                results_data["MANAGED_FLOOD"].append(entry)

    elif router_type == conf.ROUTER_TYPE.GOSSIP:
        for mr in mobility_ratios:
            for ms in  movement_speeds:
                entry = {
                    "p": gossip_p,
                    "k": gossip_k,
                    "movement_ratio": mr,
                    "movement_speed": ms,
                }

                for result_key in results:
                    entry[result_key] = results[result_key][rt_info][(mr, ms)][0]   # Just get the value since there's only 1 node size

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

# Create one figure per k value
for k in gossip_k_vals:
    fig, axes = plt.subplots(1, len(movement_speeds), figsize=(15, 5), sharey=True)
    fig.suptitle(f'Coverage (k={k})', y=1.05)

    # Plot for each movement speed
    for ax_idx, movement_speed in enumerate(movement_speeds):
        ax = axes[ax_idx]

        # Plot Managed Flood baseline
        baseline_vals = [
            reachability_dict[baselineRt][(mr, movement_speed)][0]
            for mr in mobility_ratios
        ]
        ax.plot(
            mobility_ratios, baseline_vals,
            '--', color='gray', label='Managed Flood'
        )

        # Plot GOSSIP results for this k
        for p in gossip_p_vals:
            rt_info = (conf.ROUTER_TYPE.GOSSIP, p, k)
            if rt_info not in reachability_dict:
                continue

            vals = [
                reachability_dict[rt_info][(mr, movement_speed)][0]
                for mr in mobility_ratios
            ]
            stds = [
                reachabilityStds_dict[rt_info][(mr, movement_speed)][0]
                for mr in mobility_ratios
            ]

            ax.errorbar(
                mobility_ratios, vals, yerr=stds,
                marker='o', linestyle='-',
                label=f'p={p}'
            )

        ax.set_xlabel('Mobility Ratio')
        ax.set_title(f'Speed: {movement_speed} m/min')
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(f'Coverage (%)')

    # Add legend outside subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),
        frameon=False
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Coverage_k{k}.png"), bbox_inches='tight')
    plt.close()
