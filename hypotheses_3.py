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

from lib.hypotheses_3_config import Config3
from lib.common import *
from lib.packet import *
from lib.mac import *
from lib.discrete_event import *
from lib.node import *
from lib.batch_common import *

# Debug
conf = Config3()
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
numberOfNodes = [ 5, 10, 15, 30, 75, 100 ]
gossip_p_vals = [ 0.55, 0.6, 0.65, 0.7, 0.75 ]
gossip_k_vals = [ 1, 2, 4 ]

routerTypes = [ (conf.ROUTER_TYPE.MANAGED_FLOOD, None, None) ]
for p in gossip_p_vals:
    for k in gossip_k_vals:
        routerTypes.append( (conf.ROUTER_TYPE.GOSSIP, p, k) )

output_dir = f"./out/hypotheses_3/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

collisions_dict = {}
collisionsStds_dict = {}

# Pre-Generate random positions
positions_cache = gen_random_positions(conf, repetitions, numberOfNodes)

# Initialize dictionaries for each router type, individual dictionaries for 
for rt in routerTypes:
    collisions_dict[rt] = []

###########################################################
# Main simulation loops
###########################################################

# Outer loop for each router type
for rt_i, routerType in enumerate(routerTypes):
    routerTypeLabel, gossip_p, gossip_k = routerType

    collisions = []
    collisionsStds = []

    # Inner loop for each nrNodes
    for p, nrNodes in enumerate(numberOfNodes):
        collisionRate = [ 0 for _ in range(repetitions)]

        if routerTypeLabel == conf.ROUTER_TYPE.MANAGED_FLOOD:
            print(f"\n[Router: {routerTypeLabel}] Start of {p+1} out of {len(numberOfNodes)} - {nrNodes} nodes")
        else:
            print(f"\n[Router: {routerTypeLabel}({gossip_p}, {gossip_k})] Start of {p+1} out of {len(numberOfNodes)} - {nrNodes} nodes")

        for rep in range(repetitions):
            # For the highest degree of separation between runs, config
            # should be instantiated every repetition for this router type and node number
            routerTypeConf = Config3()
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
            nrCollisions = sum([1 for pkt in packets for n in nodes if pkt.collidedAtN[n.nodeid]])
            nrSensed = sum([1 for pkt in packets for n in nodes if pkt.sensedByN[n.nodeid]])
            
            if nrSensed != 0:
                collisionRate[rep] = ( float(nrCollisions) / nrSensed ) * 100
            else:
                collisionRate[rep] = np.NaN

        # After finishing all repetitions for this nrNodes, compute means/stdevs
        collisions.append(np.nanmean(collisionRate))
        collisionsStds.append(np.nanstd(collisionRate))

    collisions_dict[routerType] = collisions
    collisionsStds_dict[routerType] = collisionsStds


###################################################
# Save to json file
###################################################
results = {
    "collisions": collisions_dict,
    "collisions_stds": collisionsStds_dict,
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


