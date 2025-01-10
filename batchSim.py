#!/usr/bin/env python3
import matplotlib
try:
	matplotlib.use("TkAgg")
except ImportError:
	print('Tkinter is needed. Install python3-tk with your package manager.')
	exit(1)

from lib.common import *
from lib.packet import *
from lib.mac import *
from lib.discrete_event import *
from lib.node import *

VERBOSE = False
SAVE = True

if VERBOSE:
	def verboseprint(*args, **kwargs): 
		print(*args, **kwargs)
else:   
	def verboseprint(*args, **kwargs): 
		pass


repetitions = 100
parameters = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25]
collisions = []
reachability = []
usefulness = []
meanDelays = []
meanTxAirUtils = []
collisionStds = []
reachabilityStds = []
usefulnessStds = []
delayStds = []
txAirUtilsStds = []
for p, nrNodes in enumerate(parameters):
	conf.NR_NODES = nrNodes
	nodeReach = [0 for _ in range(repetitions)]
	nodeUsefulness = [0 for _ in range(repetitions)]
	collisionRate = [0 for _ in range(repetitions)]
	meanDelay = [0 for _ in range(repetitions)]
	meanTxAirUtilization = [0 for _ in range(repetitions)]
	print("\nStart of", p+1, "out of", len(parameters), "value", nrNodes)
	for rep in range(repetitions):
		setBatch(rep)
		random.seed(rep)
		env = simpy.Environment()
		bc_pipe = BroadcastPipe(env)

		nodes = []
		messages = []
		packets = []
		delays = []
		packetsAtN = [[] for _ in range(conf.NR_NODES)]
		messageSeq = {"val": 0}

		found = False
		while not found:
			nodes = []
			for nodeId in range(conf.NR_NODES):
				node = MeshNode(nodes, env, bc_pipe, nodeId, conf.PERIOD, messages, packetsAtN, packets, delays, None, messageSeq)
				if node.x == None:
					break
				nodes.append(node)
			if len(nodes) == conf.NR_NODES:
				found = True

		# start simulation
		env.run(until=conf.SIMTIME)
		nrCollisions = sum([1 for p in packets for n in nodes if p.collidedAtN[n.nodeid] == True])
		nrSensed = sum([1 for p in packets for n in nodes if p.sensedByN[n.nodeid] == True])
		nrReceived = sum([1 for p in packets for n in nodes if p.receivedAtN[n.nodeid] == True])
		nrUseful = sum([n.usefulPackets for n in nodes])
		if nrSensed != 0:
			collisionRate[rep] = float((nrCollisions)/nrSensed)*100
		else:
			collisionRate[rep] = np.NaN
		if messageSeq["val"] != 0: 
			nodeReach[rep] = nrUseful/(messageSeq["val"]*(conf.NR_NODES-1))*100
		else: 
			nodeReach[rep] = np.NaN
		if nrReceived != 0:
			nodeUsefulness[rep] = nrUseful/nrReceived*100  # nr of packets that delivered to a message to a new receiver out of all packets received
		else: 
			nodeUsefulness[rep] = np.NaN
		meanDelay[rep] = np.nanmean(delays)
		meanTxAirUtilization[rep] = sum([n.txAirUtilization for n in nodes])/conf.NR_NODES
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
			"MODEM": conf.NR_NODES, 
			"MODEL": conf.MODEL,
			"NR_NODES": conf.NR_NODES,
			"INTERFERENCE_LEVEL": conf.INTERFERENCE_LEVEL,
			"COLLISION_DUE_TO_INTERFERENCE": conf.COLLISION_DUE_TO_INTERFERENCE,
			"XSIZE": conf.XSIZE, 
			"YSIZE": conf.YSIZE,
			"MINDIST": conf.MINDIST,
			"SIMTIME": conf.SIMTIME,
			"PERIOD": conf.PERIOD,
			"PACKETLENGTH": conf.PACKETLENGTH,  
			"nrMessages": messageSeq["val"]
		}
		subdir = "hopLimit3"
		simReport(data, subdir, nrNodes)
	print('Collision rate average:', round(np.nanmean(collisionRate), 2))
	print('Reachability average:', round(np.nanmean(nodeReach), 2))
	print('Usefulness average:', round(np.nanmean(nodeUsefulness), 2))
	print('Delay average:', round(np.nanmean(meanDelay), 2))
	print('Tx air utilization average:', round(np.nanmean(meanTxAirUtilization), 2))
	collisions.append(np.nanmean(collisionRate))
	reachability.append(np.nanmean(nodeReach))
	usefulness.append(np.nanmean(nodeUsefulness))
	meanDelays.append(np.nanmean(meanDelay))
	meanTxAirUtils.append(np.nanmean(meanTxAirUtilization))
	collisionStds.append(np.nanstd(collisionRate))
	reachabilityStds.append(np.nanstd(nodeReach))
	usefulnessStds.append(np.nanstd(nodeUsefulness))
	delayStds.append(np.nanstd(meanDelay))
	txAirUtilsStds.append(np.nanstd(meanTxAirUtilization))


plt.errorbar(parameters, collisions, collisionStds, fmt='-o', capsize=3, ecolor='red', elinewidth=0.5, capthick=0.5)
plt.xlabel('#nodes')
plt.ylabel('Collision rate (%)')
plt.figure()
plt.errorbar(parameters, meanDelays, delayStds, fmt='-o', capsize=3, ecolor='red', elinewidth=0.5, capthick=0.5)
plt.xlabel('#nodes')
plt.ylabel('Average delay (ms)')
plt.figure()
plt.errorbar(parameters, meanTxAirUtils, txAirUtilsStds, fmt='-o', capsize=3, ecolor='red', elinewidth=0.5, capthick=0.5)
plt.xlabel('#nodes')
plt.ylabel('Average Tx air utilization (ms)')
plt.figure()
plt.errorbar(parameters, reachability, reachabilityStds, fmt='-o', capsize=3, ecolor='red', elinewidth=0.5, capthick=0.5)
plt.xlabel('#nodes')
plt.ylabel('Reachability (%)')
plt.figure()
plt.errorbar(parameters, usefulness, usefulnessStds, fmt='-o', capsize=3, ecolor='red', elinewidth=0.5, capthick=0.5)
plt.xlabel('#nodes')
plt.ylabel('Usefulness (%)')
plt.show()
