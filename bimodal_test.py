#!/usr/bin/env python3
import collections
import time
import matplotlib
import sys, os
import seaborn as sns
import json

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
from lib.phy import *


###########################################################
# Histogram
###########################################################
def plot_coverage(coverage):
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    plt.figure(figsize=(8, 6))
    sns.histplot(
        coverage,
        stat='probability',
        bins = 10,
        binrange=(0, 1),
        kde=False
    )
    plt.margins(x=0)
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.xlabel('Fraction of nodes receiving the message', fontsize=14)
    plt.ylabel('Fraction of Executions', fontsize=14)
    plt.title(f'Distribution of Reachability in GOSSIP1({gossip_p}, {gossip_k})', fontsize=16)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.savefig(os.path.join(output_dir, 'bimodal_test.png'))
    plt.show()


# TODO - There should really be two separate concepts here, a STATE and a CONFIG
# today, the config also maintains state
conf = Config()

VERBOSE = False
SHOW_GRAPH = False

repetitions = 120
numberOfNodes = [ 100 ]
gossip_p = float(sys.argv[1])
gossip_k = int(sys.argv[2])

output_dir = f"./out/test_results/bimodal_test/bimodal_test_{gossip_p}_{gossip_k}/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if len(sys.argv) == 4 and sys.argv[3] == "-g":
    try:
        with open(os.path.join(output_dir, "data.json"), "r") as data:
            coverage_data = json.load(data)
            plot_coverage(coverage_data)
            sys.exit(0)
    except FileNotFoundError:
        print(f"No data for those p={gossip_p} and k={gossip_k} exist.")
        sys.exit(-1)

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

##############################################################################
# Pre generate node positions so we have apples to apples between router types
##############################################################################
class TempNode:
    """A lightweight node-like object with .x and .y attributes."""
    def __init__(self, x, y):
        self.x = x
        self.y = y

positions_cache = {}  # (nrNodes, rep) -> list of (x, y)

"""
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
"""

cols = 20
rows = 50

scale_range = 0.9
node_space = phy.MAXRANGE * scale_range

grid_width = cols * node_space
grid_height = rows * node_space

# Generate coordinates of a grid, centered at 0, 0
grid_positions = [ ( (x - (cols - 1) / 2) * node_space, (y - (rows - 1) / 2) * node_space ) for x in range(cols) for y in range(rows) ]

###########################################################
# Main simulation loops
###########################################################

# Inner loop for each nrNodes
coverage = [0 for _ in range(repetitions)]

for rep in range(repetitions):
    # For the highest degree of separation between runs, config
    # should be instantiated every repetition for this router type and node number
    routerTypeConf = Config()
    routerTypeConf.SELECTED_ROUTER_TYPE = routerTypeConf.ROUTER_TYPE.GOSSIP
    #routerTypeConf.NR_NODES = numberOfNodes[0]
    routerTypeConf.NR_NODES = cols * rows

    routerTypeConf.GOSSIP_P = gossip_p
    routerTypeConf.GOSSIP_K = gossip_k

    routerTypeConf.updateRouterDependencies()

    effectiveSeed = 10000 + rep
    routerTypeConf.SEED = effectiveSeed
    random.seed(effectiveSeed)

    env = simpy.Environment()
    bc_pipe = BroadcastPipe(env)

    # Start the progress-logging process
    env.process(simulationProgress(env, rep, repetitions, routerTypeConf.SIMTIME))

    # Retrieve the pre-generated positions for this (nrNodes, rep)
    #coords = positions_cache[(nrNodes, rep)]

    nodes = []
    messages = []
    packets = []
    delays = []
    packetsAtN = [[] for _ in range(routerTypeConf.NR_NODES)]
    messageSeq = {"val": 0}

    if SHOW_GRAPH:
        graph = Graph(routerTypeConf)

    for nodeId in range(routerTypeConf.NR_NODES):
        x, y = grid_positions[nodeId]

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
            messageSeq, verboseprint, False, rep
        )
        nodes.append(node)
        if SHOW_GRAPH:
            graph.addNode(node)

    if routerTypeConf.MOVEMENT_ENABLED and SHOW_GRAPH:
        env.process(runGraphUpdates(env, graph, nodes))

    totalPairs, symmetricLinks, asymmetricLinks, noLinks = setupAsymmetricLinks(routerTypeConf, nodes)

    """
    # Pick a node to send a packet
    random_node = random.randint(0, routerTypeConf.NR_NODES-1)
    nodes[random_node].send_packet = True
    """

    # Same node sends everytime
    sender = 0
    nodes[sender].send_packet = True

    # Start simulation
    env.run(until=routerTypeConf.SIMTIME)

    # Calculate stats
    nrSensed = sum([1 for pkt in packets for n in nodes if pkt.sensedByN[n.nodeid]])
    nrReceived = sum([1 for pkt in packets for n in nodes if pkt.receivedAtN[n.nodeid]])
    nrUseful = sum([n.usefulPackets for n in nodes])

    num_reached = 0
    for node in nodes:
        if 1 in node.seenPackets:
            num_reached += 1

    coverage[rep] = num_reached / routerTypeConf.NR_NODES


with open(os.path.join(output_dir, "data.json"), "w") as fp:
    json.dump(coverage, fp, indent=4)

plot_coverage(coverage)

