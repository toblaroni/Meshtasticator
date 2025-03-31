#!/usr/bin/env python3
import collections
import time
import matplotlib

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

# How many times should each combination run
repetitions = 3

# How many nodes should be simulated in each test
numberOfNodes = [ 5, 10, 15, 30, 50 ]

pValues = [ 0.1, 0.3, 0.5, 0.7 ]
kValues = [ 1, 2, 3, 4, 5 ]


#######################################
####### SET BATCH PARAMS ABOVE ########
#######################################

if VERBOSE:
    def verboseprint(*args, **kwargs): 
        print(*args, **kwargs)
else:
    def verboseprint(*args, **kwargs): 
        pass

###########################################################
# Progress-logging process
###########################################################
def simulationProgress(env, currentRep, repetitions, endTime):
    """
    Keep track of the ratio of real time per sim-second over
    a fixed sliding window, so if the simulation slows down near the end,
    the time-left estimate adapts quickly.
    """
    startWallTime = time.time()
    lastWallTime = startWallTime
    lastEnvTime = env.now
    
    # We'll store the last N ratio measurements
    N = 10
    ratios = collections.deque(maxlen=N)
    
    while True:
        fraction = env.now / endTime
        fraction = min(fraction, 1.0)
        
        # Current real time
        currentWallTime = time.time()
        realTimeDelta = currentWallTime - lastWallTime
        simTimeDelta = env.now - lastEnvTime
        
        # Compute new ratio if sim actually advanced
        if simTimeDelta > 0:
            instant_ratio = realTimeDelta / simTimeDelta
            ratios.append(instant_ratio)
        
        # If we have at least one ratio, compute a 'recent average'
        if len(ratios) > 0:
            avgRatio = sum(ratios) / len(ratios)
        else:
            avgRatio = 0.0
        
        # time_left_est = avg_ratio * (endTime - env.now)
        simTimeRemaining = endTime - env.now
        timeLeftEst = simTimeRemaining * avgRatio
        
        # Format mm:ss
        minutes = int(timeLeftEst // 60)
        seconds = int(timeLeftEst % 60)
        
        print(
            f"\rSimulation {currentRep+1}/{repetitions} progress: "
            f"{fraction*100:.1f}% | ~{minutes}m{seconds}s left...",
            end="", flush=True
        )
        
        # If done or overshoot
        if fraction >= 1.0:
            break
        
        # Update references
        lastWallTime = currentWallTime
        lastEnvTime = env.now
        
        yield env.timeout(10 * conf.ONE_SECOND_INTERVAL)

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

# If you have link asymmetry metrics
asymmetricLinkRate_dict = {}
symmetricLinkRate_dict = {}
noLinkRate_dict = {}

# Initialize dictionaries for each router type
for pVal in pValues:
    collisions_dict[pVal] = []
    collisionStds_dict[pVal] = []
    meanDelays_dict[pVal] = []
    delayStds_dict[pVal] = []
    meanTxAirUtils_dict[pVal] = []
    txAirUtilsStds_dict[pVal] = []
    reachability_dict[pVal] = []
    reachabilityStds_dict[pVal] = []
    usefulness_dict[pVal] = []
    usefulnessStds_dict[pVal] = []

    asymmetricLinkRate_dict[pVal] = []
    symmetricLinkRate_dict[pVal] = []
    noLinkRate_dict[pVal] = []



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
for pv_i, pVal in enumerate(kValues):
    pValueLabel = str(pVal)

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

        print(f"\n[Router: {pValueLabel}] Start of {p+1} out of {len(numberOfNodes)} - {nrNodes} nodes")

        for rep in range(repetitions):
            # For the highest degree of separation between runs, config
            # should be instantiated every repetition for this router type and node number
            routerTypeConf = Config()
            routerTypeConf.SELECTED_ROUTER_TYPE = routerTypeConf.ROUTER_TYPE.GOSSIP
            routerTypeConf.NR_NODES = nrNodes

            routerTypeConf.GOSSIP_P = 0.5
            routerTypeConf.GOSSIP_K = pVal

            routerTypeConf.updateRouterDependencies()

            effectiveSeed = pv_i * 10000 + rep
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
                "SELECTED_ROUTER_TYPE": routerTypeConf.ROUTER_TYPE.GOSSIP
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
    collisions_dict[pVal] = collisions
    collisionStds_dict[pVal] = collisionsStds
    reachability_dict[pVal] = reachability
    reachabilityStds_dict[pVal] = reachabilityStds
    usefulness_dict[pVal] = usefulness
    usefulnessStds_dict[pVal] = usefulnessStds
    meanDelays_dict[pVal] = meanDelays
    delayStds_dict[pVal] = delayStds
    meanTxAirUtils_dict[pVal] = meanTxAirUtils
    txAirUtilsStds_dict[pVal] = txAirUtilsStds
    asymmetricLinkRate_dict[pVal] = asymmetricLinkRateAll
    symmetricLinkRate_dict[pVal] = symmetricLinkRateAll
    noLinkRate_dict[pVal] = noLinkRateAll

###########################################################
# Plotting
###########################################################

def router_type_label(rt):
    if rt == conf.ROUTER_TYPE.MANAGED_FLOOD:
        return "Managed Flood"
    else:
        return str(rt)

###########################################################
# Choose a baseline router type for comparison
###########################################################
baselinePVal = 0.5

###########################################################
# 1) Collision Rate (with annotations)
###########################################################

plt.figure()

# Plot all router types
for pVal in pValues:
    plt.errorbar(
        numberOfNodes,
        collisions_dict[pVal],
        collisionStds_dict[pVal],
        fmt='-o', capsize=3, ecolor='red', elinewidth=0.5, capthick=0.5,
        label=router_type_label(pVal)
    )

# Now annotate differences for each router type relative to the baseline
# We want small text near each data point
for pVal in pValues:
    if pVal == baselinePVal:
        # Skip annotating differences for the baseline itself
        continue

    for i, n in enumerate(numberOfNodes):
        base_val = collisions_dict[baselinePVal][i]
        pv_val   = collisions_dict[pVal][i]

        # Compute percentage difference relative to baseline
        if base_val != 0:
            pct_diff = 100.0 * (pv_val - base_val) / base_val
        else:
            pct_diff = 0.0

        # Slight offsets so text isn't directly on top of marker
        x_offset = 0.0
        y_offset = 0.5

        plt.text(
            n + x_offset, 
            pv_val + y_offset, 
            f'{pct_diff:.1f}%', 
            ha='center', 
            fontsize=8
        )

plt.xlabel('#nodes')
plt.ylabel('Collision rate (%)')
plt.legend()
plt.title('Collision Rate by Value of P (with % Diff Annotations)')

###########################################################
# 2) Average Delay (with annotations)
###########################################################

plt.figure()

for pVal in pValues:
    plt.errorbar(
        numberOfNodes,
        meanDelays_dict[pVal],
        delayStds_dict[pVal],
        fmt='-o', capsize=3, ecolor='red', elinewidth=0.5, capthick=0.5,
        label=router_type_label(pVal)
    )

# Annotate differences (relative to baseline) at each data point
for pVal in pValues:
    if pVal == baselinePVal:
        continue

    for i, n in enumerate(numberOfNodes):
        base_val = meanDelays_dict[baselinePVal][i]
        pv_val   = meanDelays_dict[pVal][i]
        if base_val != 0:
            pct_diff = 100.0 * (pv_val - base_val) / base_val
        else:
            pct_diff = 0.0

        plt.text(
            n, pv_val + 5,  # a small offset in the y-axis
            f'{pct_diff:.1f}%', 
            ha='center', 
            fontsize=8
        )

plt.xlabel('#nodes')
plt.ylabel('Average delay (ms)')
plt.legend()
plt.title('Average Delay by Value of P (with % Diff Annotations)')

###########################################################
# 3) Average Tx air utilization (with annotations)
###########################################################

plt.figure()
for pVal in pValues:
    plt.errorbar(
        numberOfNodes,
        meanTxAirUtils_dict[pVal],
        txAirUtilsStds_dict[pVal],
        fmt='-o', capsize=3, ecolor='red', elinewidth=0.5, capthick=0.5,
        label=router_type_label(pVal)
    )

for pVal in pValues:
    if pVal == baselinePVal:
        continue

    for i, n in enumerate(numberOfNodes):
        base_val = meanTxAirUtils_dict[pVal][i]
        pv_val   = meanTxAirUtils_dict[pVal][i]
        if base_val != 0:
            pct_diff = 100.0 * (pv_val - base_val) / base_val
        else:
            pct_diff = 0.0

        plt.text(
            n, pv_val + 1,  # small offset
            f'{pct_diff:.1f}%', 
            ha='center', 
            fontsize=8
        )

plt.xlabel('#nodes')
plt.ylabel('Average Tx air utilization (ms)')
plt.legend()
plt.title('Tx Air Utilization by P Value (with % Diff Annotations)')

###########################################################
# 4) Reachability (with annotations)
###########################################################

plt.figure()
for pVal in pValues:
    plt.errorbar(
        numberOfNodes,
        reachability_dict[pVal],
        reachabilityStds_dict[pVal],
        fmt='-o', capsize=3, ecolor='red', elinewidth=0.5, capthick=0.5,
        label=router_type_label(pVal)
    )

for pVal in pValues:
    if pVal == baselinePVal:
        continue

    for i, n in enumerate(numberOfNodes):
        base_val = reachability_dict[baselinePVal][i]
        pv_val   = reachability_dict[pVal][i]
        if base_val != 0:
            pct_diff = 100.0 * (pv_val - base_val) / base_val
        else:
            pct_diff = 0.0

        plt.text(
            n, pv_val + 0.5,
            f'{pct_diff:.1f}%', 
            ha='center', 
            fontsize=8
        )

plt.xlabel('#nodes')
plt.ylabel('Reachability (%)')
plt.legend()
plt.title('Reachability by P Value (with % Diff Annotations)')

###########################################################
# 5) Usefulness (with annotations)
###########################################################

plt.figure()
for pVal in pValues:
    plt.errorbar(
        numberOfNodes,
        usefulness_dict[pVal],
        usefulnessStds_dict[pVal],
        fmt='-o', capsize=3, ecolor='red', elinewidth=0.5, capthick=0.5,
        label=router_type_label(pVal)
    )

for pVal in pValues:
    if pVal == baselinePVal:
        continue

    for i, n in enumerate(numberOfNodes):
        base_val = usefulness_dict[baselinePVal][i]
        pv_val   = usefulness_dict[pVal][i]
        if base_val != 0:
            pct_diff = 100.0 * (pv_val - base_val) / base_val
        else:
            pct_diff = 0.0

        plt.text(
            n, pv_val + 0.5,
            f'{pct_diff:.1f}%', 
            ha='center', 
            fontsize=8
        )

plt.xlabel('#nodes')
plt.ylabel('Usefulness (%)')
plt.legend()
plt.title('Usefulness by Value of P (with % Diff Annotations)')

plt.figure()

###########################################################
# 6) Show all the plots at once
###########################################################
plt.show()
