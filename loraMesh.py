#!/usr/bin/env python3
import sys

from lib.common import *
from lib.discrete_event import *
from lib.mac import *
from lib.packet import *
from lib import config as conf

VERBOSE = True
random.seed(conf.SEED)

class MeshNode():
	def __init__(self, nodes, env, bc_pipe, nodeid, period, messages, packetsAtN, packets, delays, nodeConfig):
		self.nodeid = nodeid
		if nodeConfig is not None: 
			self.x = nodeConfig['x']
			self.y = nodeConfig['y']
			self.z = nodeConfig['z']
			self.isRouter = nodeConfig['isRouter']
			self.isRepeater = nodeConfig['isRepeater']
			self.isClientMute = nodeConfig['isClientMute']
			self.hopLimit = nodeConfig['hopLimit']
			self.antennaGain = nodeConfig['antennaGain']
		else: 
			self.x, self.y = findRandomPosition(nodes)
			self.z = conf.HM
			self.isRouter = conf.router
			self.isRepeater = False
			self.isClientMute = False
			self.hopLimit = conf.hopLimit
			self.antennaGain = conf.GL
		self.messageSeq = messageSeq
		self.env = env
		self.period = period
		self.bc_pipe = bc_pipe
		self.rx_snr = 0
		self.nodes = nodes
		self.messages = messages
		self.packetsAtN = packetsAtN
		self.nrPacketsSent = 0
		self.packets = packets
		self.delays = delays
		self.leastReceivedHopLimit = {}
		self.isReceiving = []
		self.isTransmitting = False
		self.usefulPackets = 0
		self.txAirUtilization = 0
		self.airUtilization = 0
		self.droppedByDelay = 0
		self.droppedByCoverage = 0
		self.coverageBeforeDrop = 0
		self.rebroadcastPackets = 0
		self.hasReceivedAnyPacket = False
		self.coverageKnowledge = set()
		self.lastHeardTime = {}
		self.isMoving = False
		self.distanceMoved = 0

		if not self.isRepeater:  # repeaters don't generate messages themselves
			env.process(self.generateMessage())
		env.process(self.receive(self.bc_pipe.get_output_conn()))
		self.transmitter = simpy.Resource(env, 1)

		# start mobility if enabled
		if conf.MOVEMENT_ENABLED and random.random() <= conf.APPROX_RATIO_NODES_MOVING:
			self.isMoving = True
			env.process(self.moveNode(env))

	def updateCoverageKnowledge(self, neighbor_id):
		self.coverageKnowledge.add(neighbor_id)
		self.lastHeardTime[neighbor_id] = self.env.now

	def removeStaleNodes(self):
		# remove nodes we haven't been heard from in X seconds
		for n in list(self.coverageKnowledge):
			if (self.env.now - self.lastHeardTime[n]) > conf.RECENCY_THRESHOLD_SECONDS:
				self.coverageKnowledge.remove(n)
				del self.lastHeardTime[n]

	def getCoverageKnowledge(self):
		# force a stale cleanup first
		self.removeStaleNodes()
		return self.coverageKnowledge

	def moveNode(self, env):
		while True:
			# Pick a random direction and distance
			angle = 2 * math.pi * random.random()
			distance = conf.MOVEMENT_STEP_SIZE * random.random()
			
			# Compute new position
			dx = distance * math.cos(angle)
			dy = distance * math.sin(angle)
			
			leftBound   = conf.OX - conf.XSIZE/2
			rightBound  = conf.OX + conf.XSIZE/2
			bottomBound = conf.OY - conf.YSIZE/2
			topBound    = conf.OY + conf.YSIZE/2

			# Then in moveNode:
			new_x = min(max(self.x + dx, leftBound), rightBound)
			new_y = min(max(self.y + dy, bottomBound), topBound)

			# we could also vary Z if we wanted
			self.distanceMoved += calcDist(self.x, new_x, self.y, new_y, self.z, self.z) 
			
			# Update node’s position
			self.x = new_x
			self.y = new_y
			
			# Wait until next move
			yield env.timeout(conf.MOVEMENT_DELAY)

	def generateMessage(self):
		global messageSeq
		while True:
			nextGen = random.expovariate(1.0/float(self.period))
			# do not generate message near the end of the simulation (otherwise flooding cannot finish in time)
			if self.env.now+nextGen+self.hopLimit*airtime(conf.SFMODEM[conf.MODEM], conf.CRMODEM[conf.MODEM], conf.PACKETLENGTH, conf.BWMODEM[conf.MODEM]) < conf.SIMTIME:
				yield self.env.timeout(nextGen) 

				if conf.DMs:
					destId = random.choice([i for i in range(0, len(nodes)) if i is not self.nodeid])
				else:
					destId = NODENUM_BROADCAST
				messageSeq += 1
				self.messages.append(MeshMessage(self.nodeid, destId, self.env.now, messageSeq))
				p = MeshPacket(self.nodes, self.nodeid, destId, self.nodeid, conf.PACKETLENGTH, messageSeq, self.env.now, True, False, None)  
				verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'generated message', p.seq, 'to', destId)
				self.packets.append(p)
				self.env.process(self.transmit(p))
				while p.wantAck: # ReliableRouter: retransmit message if no ACK received after timeout 
					retransmissionMsec = getRetransmissionMsec(self, p) 
					yield self.env.timeout(retransmissionMsec)

					ackReceived = False  # check whether you received an ACK on the transmitted message
					minRetransmissions = conf.maxRetransmission
					for packetSent in self.packets:
						if packetSent.origTxNodeId == self.nodeid and packetSent.seq == p.seq:
							if packetSent.retransmissions < minRetransmissions:
								minRetransmissions = packetSent.retransmissions
							if packetSent.ackReceived:
								ackReceived = True
					if ackReceived: 
						verboseprint('Node', self.nodeid, 'received ACK on generated message with seq. nr.', p.seq)
						break
					else: 
						if minRetransmissions > 0:  # generate new packet with same sequence number
							pNew = MeshPacket(self.nodes, self.nodeid, p.destId, self.nodeid, p.packetLen, p.seq, p.genTime, p.wantAck, False, None)  
							pNew.retransmissions = minRetransmissions-1
							verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'wants to retransmit its generated packet to', destId, 'with seq.nr.', p.seq, 'minRetransmissions', minRetransmissions)
							self.packets.append(pNew)
							self.env.process(self.transmit(pNew))
						else:
							verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'reliable send of', p.seq, 'failed.')
							break
			else:  # do not send this message anymore, since it is close to the end of the simulation
				break


	def transmit(self, packet):
		with self.transmitter.request() as request:
			yield request

			# listen-before-talk from src/mesh/RadioLibInterface.cpp 
			txTime = setTransmitDelay(self, packet) 
			verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'picked wait time', txTime)
			yield self.env.timeout(txTime)

			# wait when currently receiving or transmitting, or channel is active
			while any(self.isReceiving) or self.isTransmitting or isChannelActive(self, self.env):
				verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'is busy Tx-ing', self.isTransmitting, 'or Rx-ing', any(self.isReceiving), 'else channel busy!')
				txTime = setTransmitDelay(self, packet) 
				yield self.env.timeout(txTime)
			verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'ends waiting')

			# check if you received an ACK for this message in the meantime
			if packet.seq not in self.leastReceivedHopLimit:
				self.leastReceivedHopLimit[packet.seq] = packet.hopLimit+1 
			if self.leastReceivedHopLimit[packet.seq] > packet.hopLimit:  # no ACK received yet, so may start transmitting 
				verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'started low level send', packet.seq, 'hopLimit', packet.hopLimit, 'original Tx', packet.origTxNodeId)
				self.nrPacketsSent += 1
				for rx_node in self.nodes:
					if packet.sensedByN[rx_node.nodeid] == True:
						if (checkcollision(self.env, packet, rx_node.nodeid, self.packetsAtN) == 0):
							self.packetsAtN[rx_node.nodeid].append(packet)
				packet.startTime = self.env.now
				packet.endTime = self.env.now + packet.timeOnAir
				self.txAirUtilization += packet.timeOnAir
				self.airUtilization += packet.timeOnAir
				self.bc_pipe.put(packet)
				self.isTransmitting = True
				yield self.env.timeout(packet.timeOnAir)
				self.isTransmitting = False
			else:  # received ACK: abort transmit, remove from packets generated 
				verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'in the meantime received ACK, abort packet with seq. nr', packet.seq)
				self.packets.remove(packet)


	def receive(self, in_pipe):
		global messageSeq
		while True:
			p = yield in_pipe.get()
			if p.sensedByN[self.nodeid] and not p.collidedAtN[self.nodeid] and p.onAirToN[self.nodeid]:  # start of reception
				if not self.hasReceivedAnyPacket:
					self.hasReceivedAnyPacket = True

				# Update knowledge of node based on reception of packet
				self.updateCoverageKnowledge(p.txNodeId)

				if not self.isTransmitting:
					verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'started receiving packet', p.seq, 'from', p.txNodeId)
					p.onAirToN[self.nodeid] = False 
					self.isReceiving.append(True)
				else:  # if you were currently transmitting, you could not have sensed it
					verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'was transmitting, so could not receive packet', p.seq)
					p.sensedByN[self.nodeid] = False
					p.onAirToN[self.nodeid] = False
			elif p.sensedByN[self.nodeid]:  # end of reception
				try: 
					self.isReceiving[self.isReceiving.index(True)] = False 
				except: 
					pass
				self.airUtilization += p.timeOnAir
				if p.collidedAtN[self.nodeid]:
					verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'could not decode packet.')
					continue
				p.receivedAtN[self.nodeid] = True
				verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'received packet', p.seq, 'with delay', round(env.now-p.genTime, 2))
				delays.append(env.now-p.genTime)

				# update hopLimit for this message
				if p.seq not in self.leastReceivedHopLimit:  # did not yet receive packet with this seq nr.
					# verboseprint('Node', self.nodeid, 'received packet nr.', p.seq, 'orig. Tx', p.origTxNodeId, "for the first time.")
					self.usefulPackets += 1
					self.leastReceivedHopLimit[p.seq] = p.hopLimit
				if p.hopLimit < self.leastReceivedHopLimit[p.seq]:  # hop limit of received packet is lower than previously received one
					self.leastReceivedHopLimit[p.seq] = p.hopLimit

				# check if implicit ACK for own generated message
				if p.origTxNodeId == self.nodeid:
					if p.isAck:
						verboseprint('Node', self.nodeid, 'received real ACK on generated message.')
					else:
						verboseprint('Node', self.nodeid, 'received implicit ACK on message sent.')
					p.ackReceived = True
					continue

				ackReceived = False
				realAckReceived = False
				for sentPacket in self.packets:
					# check if ACK for message you currently have in queue
					if sentPacket.txNodeId == self.nodeid and sentPacket.seq == p.seq:
						verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'received implicit ACK for message in queue.')
						ackReceived = True
						sentPacket.ackReceived = True
					# check if real ACK for message sent
					if sentPacket.origTxNodeId == self.nodeid and p.isAck and sentPacket.seq == p.requestId:
						verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'received real ACK.')
						realAckReceived = True
						sentPacket.ackReceived = True

				# send real ACK if you are the destination and you did not yet send the ACK
				if p.wantAck and p.destId == self.nodeid and not any(pA.requestId == p.seq for pA in self.packets):
					verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'sends a flooding ACK.')
					messageSeq += 1
					self.messages.append(MeshMessage(self.nodeid, p.origTxNodeId, self.env.now, messageSeq))
					pAck = MeshPacket(self.nodes, self.nodeid, p.origTxNodeId, self.nodeid, conf.ACKLENGTH, messageSeq, env.now, False, True, p.seq) 
					self.packets.append(pAck)
					self.env.process(self.transmit(pAck))
				# Rebroadcasting Logic for received message. This is a broadcast or a DM not meant for us.
				elif not p.destId == self.nodeid and not ackReceived and not realAckReceived and p.hopLimit > 0:
					# FloodingRouter: rebroadcast received packet
					if conf.SELECTED_ROUTER_TYPE == conf.ROUTER_TYPE.MANAGED_FLOOD:
						if not self.isClientMute:
							verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'rebroadcasts received packet', p.seq)
							pNew = MeshPacket(self.nodes, p.origTxNodeId, p.destId, self.nodeid, p.packetLen, p.seq, p.genTime, p.wantAck, False, None) 
							pNew.hopLimit = p.hopLimit-1
							self.packets.append(pNew)
							self.env.process(self.transmit(pNew))
					# BloomRouter: rebroadcast received packet
					elif conf.SELECTED_ROUTER_TYPE == conf.ROUTER_TYPE.BLOOM:
						verboseprint('Packet', p.seq, 'received at node', self.nodeid, 'with coverage', p.coverageFilter)
						pNew = MeshPacket(self.nodes, p.origTxNodeId, p.destId, self.nodeid, p.packetLen, p.seq, p.genTime, p.wantAck, False, None, p.coverageFilter, p.totalNodesInCoverageFilter)
						pNew.hopLimit = p.hopLimit-1

						# In the latest firmware, a node without any packets will always rebroadcast
						if not self.hasReceivedAnyPacket:
							rebroadcastProbability = 1.0
						else:
							rebroadcastProbability = pNew.getRebroadcastProbability()

						rebroadcastProbabilityTest = random.random()

						# Check the random against the probability
						if rebroadcastProbabilityTest <= rebroadcastProbability:
							self.rebroadcastPackets += 1
							self.packets.append(pNew)
							self.env.process(self.transmit(pNew))
							verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'rebroadcasts received packet', p.seq, 'New Coverage:', pNew.additionalCoverageRatio, 'Rnd:', rebroadcastProbabilityTest, 'Prob:', rebroadcastProbability)
						else:
							verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'drops received packet due to coverage', p.seq, 'New Coverage:', pNew.additionalCoverageRatio, 'Rnd:', rebroadcastProbabilityTest, 'Prob:', rebroadcastProbability)
							self.droppedByCoverage += 1
							self.coverageBeforeDrop += p.totalNodesInCoverageFilter
				else:
					self.droppedByDelay += 1

if VERBOSE:
	def verboseprint(*args, **kwargs): 
		print(*args, **kwargs)
else:   
	def verboseprint(*args, **kwargs): 
		pass

def runGraphUpdates(env, graph, nodes, interval=500):
    while True:
        # Wait 'interval' sim-seconds
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
messageSeq = 0
asymmetricPairs = 0

if conf.SELECTED_ROUTER_TYPE == conf.ROUTER_TYPE.BLOOM and conf.SHOW_PROBABILITY_FUNCTION_COMPARISON == True:
	plotRebroadcastProbabilityModels()

graph = Graph()
for i in range(conf.NR_NODES):
	node = MeshNode(nodes, env, bc_pipe, i, conf.PERIOD, messages, packetsAtN, packets, delays, nodeConfig[i])
	nodes.append(node)
	graph.addNode(node)
	for b in range(conf.NR_NODES):
		if i != b:
			if conf.MODEL_ASYMMETRIC_LINKS:
				conf.LINK_OFFSET[(i,b)] = random.gauss(conf.MODEL_ASYMMETRIC_LINKS_MEAN, conf.MODEL_ASYMMETRIC_LINKS_STDDEV)
			else:
				conf.LINK_OFFSET[(i,b)] = 0

for a in range(conf.NR_NODES):
	for b in range(conf.NR_NODES):
		if a != b:
			# Calculate constant RSSI in both directions
			nodeA = nodes[a]
			nodeB = nodes[b]
			distAB = calcDist(nodeA.x, nodeB.x, nodeA.y, nodeB.y, nodeA.z, nodeB.z)
			pathLossAB = estimatePathLoss(distAB, conf.FREQ, nodeA.z, nodeB.z)
			
			offsetAB = conf.LINK_OFFSET[(a, b)]
			offsetBA = conf.LINK_OFFSET[(b, a)]
			
			rssiAB = conf.PTX + nodeA.antennaGain + nodeB.antennaGain - pathLossAB - offsetAB
			rssiBA = conf.PTX + nodeB.antennaGain + nodeA.antennaGain - pathLossAB - offsetBA

			canAhearB = (rssiAB >= conf.SENSMODEM[conf.MODEM])
			canBhearA = (rssiBA >= conf.SENSMODEM[conf.MODEM])

			if (canAhearB and not canBhearA) or (not canAhearB and canBhearA):
				asymmetricPairs += 1


env.process(runGraphUpdates(env, graph, nodes))

# start simulation
print("\n====== START OF SIMULATION ======")
env.run(until=conf.SIMTIME)

# compute statistics
print("\n====== END OF SIMULATION ======")
print("\n*******************************")
print(f"\nRouter Type: {conf.SELECTED_ROUTER_TYPE}")
print('Number of messages created:', messageSeq)
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
nodeReach = sum([n.usefulPackets for n in nodes])/(messageSeq*(conf.NR_NODES-1))
print("Average percentage of nodes reached:", round(nodeReach*100, 2))
if nrReceived != 0:
	usefulness = sum([n.usefulPackets for n in nodes])/nrReceived  # nr of packets that delivered to a packet to a new receiver out of all packets sent
	print("Percentage of received packets containing new message:", round(usefulness*100, 2))
else:
	print('No packets received.')
delayDropped = sum(n.droppedByDelay for n in nodes)
print("Number of packets dropped by delay/hop limit:", delayDropped)
coverageDropped = sum(n.droppedByCoverage for n in nodes)
print("Number of packets dropped by coverage:", coverageDropped)
bloomRebroadcasts = sum(n.rebroadcastPackets for n in nodes)
avgCoverageBeforeDrop = 0
if bloomRebroadcasts > 0:
	avgCoverageBeforeDrop = float(sum(n.coverageBeforeDrop for n in nodes)) / float(bloomRebroadcasts)
print('Average Nodes in Coverage Filter Before Drop:', round(avgCoverageBeforeDrop, 2))
estimatedCoverageFPR = (1 - (1 - 1/conf.BLOOM_FILTER_SIZE_BITS)**(2 * avgCoverageBeforeDrop))**2
print("Est. Coverage Filter FPR:", round(estimatedCoverageFPR*100, 2), '%')
print("Asymmetric links modeled:", asymmetricPairs)
movingNodes = sum([1 for n in nodes if n.isMoving == True])
print("Number of moving nodes:", movingNodes)
totalDistanceMoved = sum([n.distanceMoved for n in nodes])
print("Total distance moved (km):", round(totalDistanceMoved / 1000, 0))
graph.save()

if conf.PLOT:
	plotSchedule(packets, messages)
