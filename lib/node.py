#!/usr/bin/env python3
import sys

from lib.common import *
from lib.discrete_event import *
from lib.mac import *
from lib.packet import *
from lib import config as conf


class MeshNode():
	def __init__(self, nodes, env, bc_pipe, nodeid, period, messages, packetsAtN, packets, delays, nodeConfig, messageSeq):
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
		self.coverageFalsePositives = 0
		self.coverageFalseNegatives = 0
		self.coverageKnowledge = set()
		self.lastHeardTime = {}
		self.isMoving = False
		# Track last broadcast position/time
		self.lastBroadcastX = self.x
		self.lastBroadcastY = self.y
		self.lastBroadcastTime = 0
		# track total transmit time for the last 6 buckets (each is 10s in firmware logic)
		self.channelUtilization = [0]*conf.CHANNEL_UTILIZATION_PERIODS  # each entry is ms spent on air in that interval
		self.channelUtilizationIndex = 0  # which "bucket" is current
		self.prevTxAirUtilization = 0.0   # how much total tx air-time had been used at last sample

		env.process(self.trackChannelUtilization(env))
		if not self.isRepeater:  # repeaters don't generate messages themselves
			env.process(self.generateMessage())
		env.process(self.receive(self.bc_pipe.get_output_conn()))
		self.transmitter = simpy.Resource(env, 1)

		# start mobility if enabled
		if conf.MOVEMENT_ENABLED and random.random() <= conf.APPROX_RATIO_NODES_MOVING:
			self.isMoving = True

			# Randomly assign a movement speed
			possibleSpeeds = [
				conf.WALKING_METERS_PER_MIN,  # e.g.,  96 m/min
				conf.BIKING_METERS_PER_MIN,   # e.g., 390 m/min
				conf.DRIVING_METERS_PER_MIN   # e.g., 1500 m/min
			]
			self.movementStepSize = random.choice(possibleSpeeds)

			env.process(self.moveNode(env))

	def trackChannelUtilization(self, env):
		"""
		Periodically compute how many seconds of airtime this node consumed
		over the last 10-second block and store it in the ring buffer.
		"""
		while True:
			# Wait 10 seconds of simulated time
			yield env.timeout(conf.TEN_SECONDS_INTERVAL)

			curTotalAirtime = self.txAirUtilization  # total so far, in *milliseconds*
			blockAirtimeMs = curTotalAirtime - self.prevTxAirUtilization

			self.channelUtilization[self.channelUtilizationIndex] = blockAirtimeMs

			self.prevTxAirUtilization = curTotalAirtime
			self.channelUtilizationIndex = (self.channelUtilizationIndex + 1) % conf.CHANNEL_UTILIZATION_PERIODS

	def channelUtilizationPercent(self) -> float:
		"""
		Returns how much of the last 60 seconds (6 x 10s) this node spent transmitting, as a percent.
		"""
		sumMs = sum(self.channelUtilization)
		# 6 intervals, each 10 seconds = 60,000 ms total
		# fraction = sum_ms / 60000, then multiply by 100 for percent
		return (sumMs / (conf.CHANNEL_UTILIZATION_PERIODS * conf.TEN_SECONDS_INTERVAL)) * 100.0

	def updateCoverageKnowledge(self, neighbor_id):
		self.coverageKnowledge.add(neighbor_id)
		self.lastHeardTime[neighbor_id] = self.env.now

	def removeStaleNodes(self):
		# remove nodes we haven't been heard from in X seconds
		for n in list(self.coverageKnowledge):
			if (self.env.now - self.lastHeardTime[n]) > conf.RECENCY_THRESHOLD:
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
			distance = self.movementStepSize * random.random()
			
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
			
			# Update node’s position
			self.x = new_x
			self.y = new_y

			distanceTraveled = calcDist(self.lastBroadcastX, self.x, self.lastBroadcastY, self.y)
			timeElapsed = env.now - self.lastBroadcastTime
			if (distanceTraveled >= conf.SMART_POSITION_DISTANCE_THRESHOLD and
				timeElapsed >= conf.SMART_POSITION_DISTANCE_MIN_TIME):

				currentUtil = self.channelUtilizationPercent()
				if currentUtil < 25.0:
					self.sendPacket(NODENUM_BROADCAST, "POSITION")
					self.lastBroadcastX = self.x
					self.lastBroadcastY = self.y
					self.lastBroadcastTime = env.now
				else:
					verboseprint(f"At time {env.now} node {self.nodeid} SKIPS POSITION broadcast (util={currentUtil:.1f}% > 25%)")

			
			# Wait until next move
			nextMove = self.getNextTime(conf.ONE_MIN_INTERVAL)
			if nextMove >= 0:
				yield env.timeout(nextMove)
			else:
				break

	def sendPacket(self, destId, type=""):
		self.messageSeq += 1
		self.messages.append(MeshMessage(self.nodeid, destId, self.env.now, self.messageSeq))
		p = MeshPacket(self.nodes, self.nodeid, destId, self.nodeid, conf.PACKETLENGTH, self.messageSeq, self.env.now, True, False, None, self.env.now)
		verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'generated', type, 'message', p.seq, 'to', destId)
		self.packets.append(p)
		self.env.process(self.transmit(p))
		return p
	
	def getNextTime(self, period):
		nextGen = random.expovariate(1.0/float(period))
		# do not generate message near the end of the simulation (otherwise flooding cannot finish in time)
		if self.env.now+nextGen+self.hopLimit*airtime(conf.SFMODEM[conf.MODEM], conf.CRMODEM[conf.MODEM], conf.PACKETLENGTH, conf.BWMODEM[conf.MODEM]) < conf.SIMTIME:
			return nextGen
		return -1

	def generateMessage(self):
		while True:
			# Returns -1 if we won't make it before the sim ends
			nextGen = self.getNextTime(self.period)
			# do not generate message near the end of the simulation (otherwise flooding cannot finish in time)
			if nextGen >= 0:
				yield self.env.timeout(nextGen) 

				if conf.DMs:
					destId = random.choice([i for i in range(0, len(self.nodes)) if i is not self.nodeid])
				else:
					destId = NODENUM_BROADCAST

				p = self.sendPacket(destId)

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
							pNew = MeshPacket(self.nodes, self.nodeid, p.destId, self.nodeid, p.packetLen, p.seq, p.genTime, p.wantAck, False, None, self.env.now)
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
		while True:
			p = yield in_pipe.get()
			if p.sensedByN[self.nodeid] and not p.collidedAtN[self.nodeid] and p.onAirToN[self.nodeid]:  # start of reception
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
				verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'received packet', p.seq, 'with delay', round(self.env.now-p.genTime, 2))
				self.delays.append(self.env.now-p.genTime)

				# Update knowledge of node based on reception of packet
				# We only want this to be our direct neighbors because there is no other mechanism
				# in the simulator to test that
				self.updateCoverageKnowledge(p.txNodeId)

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
					self.messageSeq += 1
					self.messages.append(MeshMessage(self.nodeid, p.origTxNodeId, self.env.now, self.messageSeq))
					pAck = MeshPacket(self.nodes, self.nodeid, p.origTxNodeId, self.nodeid, conf.ACKLENGTH, self.messageSeq, self.env.now, False, True, p.seq, self.env.now) 
					self.packets.append(pAck)
					self.env.process(self.transmit(pAck))
				# Rebroadcasting Logic for received message. This is a broadcast or a DM not meant for us.
				elif not p.destId == self.nodeid and not ackReceived and not realAckReceived and p.hopLimit > 0:
					# FloodingRouter: rebroadcast received packet
					if conf.SELECTED_ROUTER_TYPE == conf.ROUTER_TYPE.MANAGED_FLOOD:
						if not self.isClientMute:
							verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'rebroadcasts received packet', p.seq)
							pNew = MeshPacket(self.nodes, p.origTxNodeId, p.destId, self.nodeid, p.packetLen, p.seq, p.genTime, p.wantAck, False, None, self.env.now) 
							pNew.hopLimit = p.hopLimit-1
							self.packets.append(pNew)
							self.env.process(self.transmit(pNew))
					# BloomRouter: rebroadcast received packet
					elif conf.SELECTED_ROUTER_TYPE == conf.ROUTER_TYPE.BLOOM:
						verboseprint('Packet', p.seq, 'received at node', self.nodeid, 'with coverage', p.coverageFilter)
						pNew = MeshPacket(self.nodes, p.origTxNodeId, p.destId, self.nodeid, p.packetLen, p.seq, p.genTime, p.wantAck, False, None, self.env.now, p.coverageFilter, p.totalNodesInCoverageFilter)
						pNew.hopLimit = p.hopLimit-1

						# Record how far off our coverage knowledge is from actual coverage
						(fp, fn) = pNew.getPacketCoverageDifference(self.nodes)
						self.coverageFalsePositives += fp
						self.coverageFalseNegatives += fn

						rebroadcastProbability = pNew.getRebroadcastProbability()
						# In the latest firmware, a node without any direct neighbor knowledge will
						# rebroadcast with UNKNOWN_COVERAGE_REBROADCAST_PROBABILITY
						# This is NOT the same as looking for a coverage ratio of 0.0
						if pNew.neighbors == 0:
							rebroadcastProbability = conf.UNKNOWN_COVERAGE_REBROADCAST_PROBABILITY
							verboseprint('Node', self.nodeid, 'has unknown coverage. Falling back to UKNOWN_COVERAGE_REBROADCAST_PROBABILITY')

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