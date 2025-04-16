#!/usr/bin/env python3
import collections
import time
import matplotlib
import sys, os

try:
    matplotlib.use("TkAgg")
except ImportError:
    print('Tkinter is needed. Install python3-tk with your package manager.')
    exit(1)
import simpy
import numpy as np
import random
import matplotlib.pyplot as plt

from lib.config import Config
from lib.common import *
from lib.packet import *
from lib.mac import *
from lib.discrete_event import *
from lib.node import *
from common

# TODO - There should really be two separate concepts here, a STATE and a CONFIG
# today, the config also maintains state
conf = Config()
VERBOSE = False
SHOW_GRAPH = False
SAVE = True

#######################################
####### SET BATCH PARAMS BELOW ########
#######################################

# Add your router types here
# This leaves room for new experimentation of 
# different routing algorithms
routerTypes = [conf.ROUTER_TYPE.MANAGED_FLOOD, conf.ROUTER_TYPE.GOSSIP]

# How many times should each combination run
repetitions = 3

# How many nodes should be simulated in each test
numberOfNodes = [ 250 ]

gossip_p = float(sys.argv[1])
gossip_k = int(sys.argv[2])

output_dir = f"./out/test_results/flood_vs_gossip_{gossip_p}_{gossip_k}/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#######################################
####### SET BATCH PARAMS ABOVE ########
#######################################

if VERBOSE:
    def verboseprint(*args, **kwargs): 
        print(*args, **kwargs)
else:
    def verboseprint(*args, **kwargs): 
        pass

# We will collect the metrics in dictionaries keyed by router type.
# For example: collisions_dict[ routerType ] = [list of mean collisions, one per nrNodes]
collisions_dict = {}
collisionStds_dict = {}
meanDelays_dict = {}
delayStds_dict = {}
meanTxAirUtils_dict = {}
txAirUtilsStds_dict = {}
reachability_dict = {}
reachabilityStds_dict = {}
usefulness_dict = {}
usefulnessStds_dict = {}

coverage_dict = {}
coverageStds_dict = {}

# If you have link asymmetry metrics
asymmetricLinkRate_dict = {}
symmetricLinkRate_dict = {}
noLinkRate_dict = {}

# Initialize dictionaries for each router type
for rt in routerTypes:
    collisions_dict[rt] = []
    collisionStds_dict[rt] = []
    meanDelays_dict[rt] = []
    delayStds_dict[rt] = []
    meanTxAirUtils_dict[rt] = []
    txAirUtilsStds_dict[rt] = []
    reachability_dict[rt] = []
    reachabilityStds_dict[rt] = []
    usefulness_dict[rt] = []
    usefulnessStds_dict[rt] = []
    coverage_dict[rt] = []
    coverageStds_dict[rt] = []

    asymmetricLinkRate_dict[rt] = []
    symmetricLinkRate_dict[rt] = []
    noLinkRate_dict[rt] = []



##############################################################################
# Pre generate node positions so we have apples to apples between router types
##############################################################################
class TempNode:
    """A lightweight node-like object with .x and .y attributes."""
    def __init__(self, x, y):
        self.x = x
        self.y = y

positions_cache = {}  # (nrNodes, rep) -> list of (x, y)

for nrNodes in numberOfNodes:
    for rep in range(repetitions):
        random.seed(rep)
        found = False
        temp_nodes = []

        # We attempt to place 'nrNodes' one by one using findRandomPosition,
        # but pass in a list of TempNode objects so it can do n.x, n.y
        while not found:
            temp_nodes = []
            for _ in range(nrNodes):
                xnew, ynew = findRandomPosition(conf, temp_nodes)
                if xnew is None:
                    # means we failed to place a node
                    break
                # Wrap coordinates in a TempNode
                temp_nodes.append(TempNode(xnew, ynew))

            if len(temp_nodes) == nrNodes:
                found = True
            else:
                pass

        # Convert the final TempNodes to (x, y) tuples
        coords = [(tn.x, tn.y) for tn in temp_nodes]
        positions_cache[(nrNodes, rep)] = coords

###########################################################
# Main simulation loops
###########################################################

# Outer loop for each router type
for rt_i, routerType in enumerate(routerTypes):
    routerTypeLabel = str(routerType)

    # Prepare arrays for the final plot data, one per metric
    collisions = []
    collisionsStds = []
    meanDelays = []
    delayStds = []
    meanTxAirUtils = []
    txAirUtilsStds = []
    reachability = []
    reachabilityStds = []
    usefulness = []
    usefulnessStds = []
    asymmetricLinkRateAll = []
    symmetricLinkRateAll = []
    noLinkRateAll = []

    meanCoverage = []
    coverageStds = []

    # Inner loop for each nrNodes
    for p, nrNodes in enumerate(numberOfNodes):

        nodeReach = [0 for _ in range(repetitions)]
        nodeUsefulness = [0 for _ in range(repetitions)]
        collisionRate = [0 for _ in range(repetitions)]
        meanDelay = [0 for _ in range(repetitions)]
        meanTxAirUtilization = [0 for _ in range(repetitions)]
        asymmetricLinkRate = [0 for _ in range(repetitions)]
        symmetricLinkRate = [0 for _ in range(repetitions)]
        noLinkRate = [0 for _ in range(repetitions)]
        avgCoverageReps = [ 0 for _ in range(repetitions)]

        print(f"\n[Router: {routerTypeLabel}] Start of {p+1} out of {len(numberOfNodes)} - {nrNodes} nodes")

        for rep in range(repetitions):
            # For the highest degree of separation between runs, config
            # should be instantiated every repetition for this router type and node number
            routerTypeConf = Config()
            routerTypeConf.SELECTED_ROUTER_TYPE = routerType
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
            env.process(simulationProgress(env, rep, repetitions, routerTypeConf.SIMTIME))

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
                    messageSeq, verboseprint
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
            nrCollisions = sum([1 for pkt in packets for n in nodes if pkt.collidedAtN[n.nodeid]])
            nrSensed = sum([1 for pkt in packets for n in nodes if pkt.sensedByN[n.nodeid]])
            nrReceived = sum([1 for pkt in packets for n in nodes if pkt.receivedAtN[n.nodeid]])
            nrUseful = sum([n.usefulPackets for n in nodes])

            avgCoverage = []
            for pktSeq in range(messageSeq["val"]):    # Loop through all packet IDs
                nodeCount = 0
                for node in nodes:
                    if routerType == conf.ROUTER_TYPE.MANAGED_FLOOD and pktSeq in node.leastReceivedHopLimit:    # Hasn't been seen by that node
                        nodeCount += 1
                    elif routerType == conf.ROUTER_TYPE.GOSSIP and pktSeq in node.seenPackets:
                        nodeCount += 1
                avgCoverage.append((nodeCount / routerTypeConf.NR_NODES)*100)

            avgCoverageReps[rep] = np.nanmean(avgCoverage)

            if nrSensed != 0:
                collisionRate[rep] = float(nrCollisions) / nrSensed * 100
            else:
                collisionRate[rep] = np.NaN

            if messageSeq["val"] != 0:
                nodeReach[rep] = nrUseful / (messageSeq["val"] * (routerTypeConf.NR_NODES - 1)) * 100
            else:
                nodeReach[rep] = np.NaN

            if nrReceived != 0:
                nodeUsefulness[rep] = nrUseful / nrReceived * 100
            else:
                nodeUsefulness[rep] = np.NaN

            meanDelay[rep] = np.nanmean(delays)
            meanTxAirUtilization[rep] = sum([n.txAirUtilization for n in nodes]) / routerTypeConf.NR_NODES

            if routerTypeConf.MODEL_ASYMMETRIC_LINKS:
                asymmetricLinkRate[rep] = round(asymmetricLinks / totalPairs * 100, 2)
                symmetricLinkRate[rep] = round(symmetricLinks / totalPairs * 100, 2)
                noLinkRate[rep] = round(noLinks / totalPairs * 100, 2)

        # After finishing all repetitions for this nrNodes, compute means/stdevs
        collisions.append(np.nanmean(collisionRate))
        collisionsStds.append(np.nanstd(collisionRate))
        reachability.append(np.nanmean(nodeReach))
        reachabilityStds.append(np.nanstd(nodeReach))
        usefulness.append(np.nanmean(nodeUsefulness))
        usefulnessStds.append(np.nanstd(nodeUsefulness))
        meanDelays.append(np.nanmean(meanDelay))
        delayStds.append(np.nanstd(meanDelay))
        meanTxAirUtils.append(np.nanmean(meanTxAirUtilization))
        txAirUtilsStds.append(np.nanstd(meanTxAirUtilization))
        asymmetricLinkRateAll.append(np.nanmean(asymmetricLinkRate))
        symmetricLinkRateAll.append(np.nanmean(symmetricLinkRate))
        noLinkRateAll.append(np.nanmean(noLinkRate))

        meanCoverage.append(np.nanmean(avgCoverageReps))
        coverageStds.append(np.nanstd(avgCoverageReps))

        # Saving to file if needed
        if SAVE:
            print('Saving to file...')
            data = {
                "CollisionRate": collisionRate,
                "Reachability": nodeReach,
                "Usefulness": nodeUsefulness,
                "meanDelay": meanDelay,
                "meanTxAirUtil": meanTxAirUtilization,
                "nrCollisions": nrCollisions,
                "nrSensed": nrSensed,
                "nrReceived": nrReceived,
                "usefulPackets": nrUseful,
                "MODEM": routerTypeConf.NR_NODES,
                "MODEL": routerTypeConf.MODEL,
                "NR_NODES": routerTypeConf.NR_NODES,
                "INTERFERENCE_LEVEL": routerTypeConf.INTERFERENCE_LEVEL,
                "COLLISION_DUE_TO_INTERFERENCE": routerTypeConf.COLLISION_DUE_TO_INTERFERENCE,
                "XSIZE": routerTypeConf.XSIZE,
                "YSIZE": routerTypeConf.YSIZE,
                "MINDIST": routerTypeConf.MINDIST,
                "SIMTIME": routerTypeConf.SIMTIME,
                "PERIOD": routerTypeConf.PERIOD,
                "PACKETLENGTH": routerTypeConf.PACKETLENGTH,
                "nrMessages": messageSeq["val"],
                "SELECTED_ROUTER_TYPE": routerTypeLabel
            }
            subdir = "hopLimit3"
            simReport(routerTypeConf, data, subdir, nrNodes)

        # Print summary
        print('Collision rate average:', round(np.nanmean(collisionRate), 2))
        print('Reachability average:', round(np.nanmean(nodeReach), 2))
        print('Usefulness average:', round(np.nanmean(nodeUsefulness), 2))
        print('Delay average:', round(np.nanmean(meanDelay), 2))
        print('Tx air utilization average:', round(np.nanmean(meanTxAirUtilization), 2))
        if routerTypeConf.MODEL_ASYMMETRIC_LINKS:
            print('Asymmetric Links:', round(np.nanmean(asymmetricLinkRate), 2))
            print('Symmetric Links:', round(np.nanmean(symmetricLinkRate), 2))
            print('No Links:', round(np.nanmean(noLinkRate), 2))

    # After finishing all nrNodes for the *current* router type,
    # store these lists in the dictionary so we can plot after.
    collisions_dict[routerType] = collisions
    collisionStds_dict[routerType] = collisionsStds
    reachability_dict[routerType] = reachability
    reachabilityStds_dict[routerType] = reachabilityStds
    usefulness_dict[routerType] = usefulness
    usefulnessStds_dict[routerType] = usefulnessStds
    meanDelays_dict[routerType] = meanDelays
    delayStds_dict[routerType] = delayStds
    meanTxAirUtils_dict[routerType] = meanTxAirUtils
    txAirUtilsStds_dict[routerType] = txAirUtilsStds
    asymmetricLinkRate_dict[routerType] = asymmetricLinkRateAll
    symmetricLinkRate_dict[routerType] = symmetricLinkRateAll
    noLinkRate_dict[routerType] = noLinkRateAll
    coverage_dict[routerType] = meanCoverage
    coverageStds_dict[routerType] = coverageStds

###########################################################
# Plotting
###########################################################

def router_type_label(rt):
    if rt == conf.ROUTER_TYPE.MANAGED_FLOOD:
        return "Managed Flood"
    elif rt == conf.ROUTER_TYPE.GOSSIP:
        return f"GOSSIP({gossip_p}, {gossip_k})"
    else:
        return str(rt)

###########################################################
# Choose a baseline router type for comparison
###########################################################
baselineRt = conf.ROUTER_TYPE.MANAGED_FLOOD

###########################################################
# 1) Collision Rate (with annotations)
###########################################################

plt.figure()

# Plot all router types
for rt in routerTypes:
    plt.errorbar(
        numberOfNodes,
        collisions_dict[rt],
        collisionStds_dict[rt],
        fmt='-o', capsize=3, ecolor='red', elinewidth=0.5, capthick=0.5,
        label=router_type_label(rt)
    )

# Now annotate differences for each router type relative to the baseline
# We want small text near each data point
for rt in routerTypes:
    if rt == baselineRt:
        # Skip annotating differences for the baseline itself
        continue

    for i, n in enumerate(numberOfNodes):
        base_val = collisions_dict[baselineRt][i]
        rt_val   = collisions_dict[rt][i]

        # Compute percentage difference relative to baseline
        if base_val != 0:
            pct_diff = 100.0 * (rt_val - base_val) / base_val
        else:
            pct_diff = 0.0

        # Slight offsets so text isn't directly on top of marker
        x_offset = 0.0
        y_offset = 0.5

        plt.text(
            n + x_offset, 
            rt_val + y_offset, 
            f'{pct_diff:.1f}%', 
            ha='center', 
            fontsize=8
        )

plt.xlabel('#nodes')
plt.ylabel('Collision rate (%)')
plt.legend()
plt.title('Collision Rate by Router Type (with % Diff Annotations)')

plt.savefig(os.path.join(output_dir, "collision_rate.png"))
plt.close()

###########################################################
# 2) Mean delays (with annotations)
###########################################################

plt.figure()

for rt in routerTypes:
    plt.errorbar(
        numberOfNodes,
        meanDelays_dict[rt],
        delayStds_dict[rt],
        fmt='-o', capsize=3, ecolor='red', elinewidth=0.5, capthick=0.5,
        label=router_type_label(rt)
    )

# Annotate differences (relative to baseline) at each data point
for rt in routerTypes:
    if rt == baselineRt:
        continue

    for i, n in enumerate(numberOfNodes):
        base_val = meanDelays_dict[baselineRt][i]
        rt_val   = meanDelays_dict[rt][i]
        if base_val != 0:
            pct_diff = 100.0 * (rt_val - base_val) / base_val
        else:
            pct_diff = 0.0

        plt.text(
            n, rt_val + 5,  # a small offset in the y-axis
            f'{pct_diff:.1f}%', 
            ha='center', 
            fontsize=8
        )

plt.xlabel('#nodes')
plt.ylabel('Average Delay (ms)')
plt.legend()
plt.title('Average Delay Between Packets by Router Type (with % Diff Annotations)')

plt.savefig(os.path.join(output_dir, "latency.png"))
plt.close()

###########################################################
# 3) Average Tx air utilization (with annotations)
###########################################################

plt.figure()
for rt in routerTypes:
    plt.errorbar(
        numberOfNodes,
        meanTxAirUtils_dict[rt],
        txAirUtilsStds_dict[rt],
        fmt='-o', capsize=3, ecolor='red', elinewidth=0.5, capthick=0.5,
        label=router_type_label(rt)
    )

for rt in routerTypes:
    if rt == baselineRt:
        continue

    for i, n in enumerate(numberOfNodes):
        base_val = meanTxAirUtils_dict[baselineRt][i]
        rt_val   = meanTxAirUtils_dict[rt][i]
        if base_val != 0:
            pct_diff = 100.0 * (rt_val - base_val) / base_val
        else:
            pct_diff = 0.0

        plt.text(
            n, rt_val + 1,  # small offset
            f'{pct_diff:.1f}%', 
            ha='center', 
            fontsize=8
        )

plt.xlabel('#nodes')
plt.ylabel('Average Tx air utilization (ms)')
plt.legend()
plt.title('Tx Air Utilization by Router Type (with % Diff Annotations)')

plt.savefig(os.path.join(output_dir, "tx_air_util.png"))
plt.close()

###########################################################
# 4) Reachability (with annotations)
###########################################################

plt.figure()
for rt in routerTypes:
    plt.errorbar(
        numberOfNodes,
        reachability_dict[rt],
        reachabilityStds_dict[rt],
        fmt='-o', capsize=3, ecolor='red', elinewidth=0.5, capthick=0.5,
        label=router_type_label(rt)
    )

for rt in routerTypes:
    if rt == baselineRt:
        continue

    for i, n in enumerate(numberOfNodes):
        base_val = reachability_dict[baselineRt][i]
        rt_val   = reachability_dict[rt][i]
        if base_val != 0:
            pct_diff = 100.0 * (rt_val - base_val) / base_val
        else:
            pct_diff = 0.0

        plt.text(
            n, rt_val + 0.5,
            f'{pct_diff:.1f}%', 
            ha='center', 
            fontsize=8
        )

plt.xlabel('#nodes')
plt.ylabel('Reachability (%)')
plt.legend()
plt.title('Reachability by Router Type (with % Diff Annotations)')

plt.savefig(os.path.join(output_dir, "reachability.png"))
plt.close()

###########################################################
# 5) Usefulness (with annotations)
###########################################################

plt.figure()
for rt in routerTypes:
    plt.errorbar(
        numberOfNodes,
        usefulness_dict[rt],
        usefulnessStds_dict[rt],
        fmt='-o', capsize=3, ecolor='red', elinewidth=0.5, capthick=0.5,
        label=router_type_label(rt)
    )

for rt in routerTypes:
    if rt == baselineRt:
        continue

    for i, n in enumerate(numberOfNodes):
        base_val = usefulness_dict[baselineRt][i]
        rt_val   = usefulness_dict[rt][i]
        if base_val != 0:
            pct_diff = 100.0 * (rt_val - base_val) / base_val
        else:
            pct_diff = 0.0

        plt.text(
            n, rt_val + 0.5,
            f'{pct_diff:.1f}%', 
            ha='center', 
            fontsize=8
        )

plt.xlabel('#nodes')
plt.ylabel('Usefulness (%)')
plt.legend()
plt.title('Usefulness by Router Type (with % Diff Annotations)')

plt.savefig(os.path.join(output_dir, "usefulness.png"))
plt.close()

###########################################################
# 6) Coverage (Flooding)
###########################################################
plt.figure()

for rt in routerTypes:
    plt.errorbar(
        numberOfNodes,
        coverage_dict[rt],
        coverageStds_dict[rt],
        fmt='-o', capsize=3, ecolor='red', elinewidth=0.5, capthick=0.5,
        label=router_type_label(rt)
    )


for rt in routerTypes:
    if rt == baselineRt:
        continue

    for i, n in enumerate(numberOfNodes):
        base_val = coverage_dict[baselineRt][i]
        rt_val   = coverage_dict[rt][i]
        if base_val != 0:
            pct_diff = 100.0 * (rt_val - base_val) / base_val
        else:
            pct_diff = 0.0

        plt.text(
            n, rt_val + 0.5,
            f'{pct_diff:.1f}%', 
            ha='center', 
            fontsize=8
        )

plt.xlabel('#nodes')
plt.ylabel('Average Coverage of Packets (%)')
plt.legend()
plt.title('Coverage of Packets by Router Type (with % Diff Annotations)')

plt.savefig(os.path.join(output_dir, "coverage.png"))
plt.close()

###########################################################
# Show all the plots at once and save
###########################################################
plt.show()


