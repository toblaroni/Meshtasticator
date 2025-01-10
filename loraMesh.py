#!/usr/bin/env python3
import sys

from lib.common import *
from lib.discrete_event import *
from lib.mac import *
from lib.packet import *
from lib.node import *
from lib import config as conf

VERBOSE = True
random.seed(conf.SEED)

if VERBOSE:
	def verboseprint(*args, **kwargs): 
		print(*args, **kwargs)
else:   
	def verboseprint(*args, **kwargs): 
		pass

def runGraphUpdates(env, graph, nodes, interval=conf.ONE_MIN_INTERVAL/2):
    while True:
        # Wait 'interval' sim-mseconds
        yield env.timeout(interval)
        # Now update the positions in the graph
        graph.updatePositions(nodes)

nodeConfig = getParams(sys.argv)
env = simpy.Environment()
bc_pipe = BroadcastPipe(env)

# simulation variables
nodes = []
messages = []
packets = []
delays = []
packetsAtN = [[] for _ in range(conf.NR_NODES)]
messageSeq = {"val": 0}
totalPairs = 0
symmetricLinks = 0
asymmetricLinks = 0
noLinks = 0

if conf.SELECTED_ROUTER_TYPE == conf.ROUTER_TYPE.BLOOM and conf.SHOW_PROBABILITY_FUNCTION_COMPARISON == True:
	plotRebroadcastProbabilityModels()

graph = Graph()
for i in range(conf.NR_NODES):
	node = MeshNode(nodes, env, bc_pipe, i, conf.PERIOD, messages, packetsAtN, packets, delays, nodeConfig[i], messageSeq, verboseprint)
	nodes.append(node)
	graph.addNode(node)
	
totalPairs, symmetricLinks, asymmetricLinks, noLinks = setupAsymmetricLinks(nodes)

if conf.MOVEMENT_ENABLED:
	env.process(runGraphUpdates(env, graph, nodes))

# start simulation
print("\n====== START OF SIMULATION ======")
env.run(until=conf.SIMTIME)

# compute statistics
print("\n====== END OF SIMULATION ======")
print("*******************************")
print(f"\nRouter Type: {conf.SELECTED_ROUTER_TYPE}")
print('Number of messages created:', messageSeq["val"])
sent = len(packets)
if conf.DMs:
	potentialReceivers = sent
else:
	potentialReceivers = sent*(conf.NR_NODES-1)
print('Number of packets sent:', sent, 'to', potentialReceivers, 'potential receivers')
nrCollisions = sum([1 for p in packets for n in nodes if p.collidedAtN[n.nodeid] == True])
print("Number of collisions:", nrCollisions)
nrSensed = sum([1 for p in packets for n in nodes if p.sensedByN[n.nodeid] == True])
print("Number of packets sensed:", nrSensed)
nrReceived = sum([1 for p in packets for n in nodes if p.receivedAtN[n.nodeid] == True])
print("Number of packets received:", nrReceived)
meanDelay = np.nanmean(delays)
print('Delay average (ms):', round(meanDelay, 2))
txAirUtilization = sum([n.txAirUtilization for n in nodes])/conf.NR_NODES/conf.SIMTIME*100
print('Average Tx air utilization:', round(txAirUtilization, 2), '%')
if nrSensed != 0:
	collisionRate = float((nrCollisions)/nrSensed)
	print("Percentage of packets that collided:", round(collisionRate*100, 2))
else:
	print("No packets sensed.")
nodeReach = sum([n.usefulPackets for n in nodes])/(messageSeq["val"]*(conf.NR_NODES-1))
print("Average percentage of nodes reached:", round(nodeReach*100, 2))
if nrReceived != 0:
	usefulness = sum([n.usefulPackets for n in nodes])/nrReceived  # nr of packets that delivered to a packet to a new receiver out of all packets sent
	print("Percentage of received packets containing new message:", round(usefulness*100, 2))
else:
	print('No packets received.')
delayDropped = sum(n.droppedByDelay for n in nodes)
print("Number of packets dropped by delay/hop limit:", delayDropped)

if conf.SELECTED_ROUTER_TYPE == conf.ROUTER_TYPE.BLOOM:
	coverageDropped = sum(n.droppedByCoverage for n in nodes)
	print("Number of packets dropped by coverage:", coverageDropped)
	bloomRebroadcasts = sum(n.rebroadcastPackets for n in nodes)
	avgCoverageBeforeDrop = 0
	if bloomRebroadcasts > 0:
		avgCoverageBeforeDrop = float(sum(n.coverageBeforeDrop for n in nodes)) / float(bloomRebroadcasts)
	print('Average Nodes in Coverage Filter Before Drop:', round(avgCoverageBeforeDrop, 2))
	estimatedCoverageFPR = (1 - (1 - 1/conf.BLOOM_FILTER_SIZE_BITS)**(2 * avgCoverageBeforeDrop))**2
	print("Est. FPR From Bloom Saturation:", round(estimatedCoverageFPR*100, 2), '%')
	coverageFp = sum([n.coverageFalsePositives for n in nodes])
	print("I think I can cover this node, but I actually can't:", round(coverageFp / potentialReceivers * 100, 2), '%')
	coverageFn = sum([n.coverageFalseNegatives for n in nodes])
	print("I don't cover this node, but I think I can:", round(coverageFn / potentialReceivers * 100, 2), '%')

if conf.MODEL_ASYMMETRIC_LINKS == True:
	print("Asymmetric links:", round(asymmetricLinks / totalPairs * 100, 2), '%')
	print("Symmetric links:", round(symmetricLinks / totalPairs * 100, 2), '%')
	print("No links:", round(noLinks / totalPairs * 100, 2), '%')

if conf.MOVEMENT_ENABLED == True:
	movingNodes = sum([1 for n in nodes if n.isMoving == True])
	print("Number of moving nodes:", movingNodes)

graph.save()

if conf.PLOT:
	plotSchedule(packets, messages)
