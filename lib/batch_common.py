import random, time, collections, os, json
from lib.common import *

###########################################################
# Progress-logging process
###########################################################
def simulationProgress(env, conf, currentRep, repetitions, endTime):
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
def gen_random_positions(conf, repetitions, numberOfNodes):
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

    return positions_cache


##############################################################################
# Save results to JSON file for quick graphing
##############################################################################
def save_results( conf, results, routerTypes, numberOfNodes, output_dir ):
    """
    Expects results to be a list of dictionarys with routerTypes being the keys
    Expects routerTypes: Tuple( ROUTER_TYPE, p, k )
    """
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
