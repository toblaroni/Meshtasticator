from lib.common import calcDist
from lib.coverageFilter import CoverageFilter
from .phy import *


NODENUM_BROADCAST = 0xFFFFFFFF
random.seed(conf.SEED)


class MeshPacket(): 
	def __init__(self, nodes, origTxNodeId, destId, txNodeId, plen, seq, genTime, wantAck, isAck, requestId, now, coverageFilter = None, prevNodesInCovFilter = 0):
		self.origTxNodeId = origTxNodeId
		self.destId = destId
		self.txNodeId = txNodeId
		self.wantAck = wantAck
		self.isAck = isAck
		self.seq = seq
		self.requestId = requestId
		self.genTime = genTime
		self.now = now
		self.txpow = conf.PTX
		self.LplAtN = [0 for _ in range(conf.NR_NODES)]
		self.rssiAtN = [0 for _ in range(conf.NR_NODES)]
		self.sensedByN = [False for _ in range(conf.NR_NODES)]
		self.detectedByN = [False for _ in range(conf.NR_NODES)]
		self.collidedAtN = [False for _ in range(conf.NR_NODES)]
		self.receivedAtN = [False for _ in range(conf.NR_NODES)]
		self.onAirToN = [True for _ in range(conf.NR_NODES)]

		# configuration values
		self.sf = conf.SFMODEM[conf.MODEM]
		self.cr = conf.CRMODEM[conf.MODEM]
		self.bw = conf.BWMODEM[conf.MODEM]
		self.freq = conf.FREQ
		self.tx_node = next(n for n in nodes if n.nodeid == self.txNodeId)
		for rx_node in nodes:
			if rx_node.nodeid == self.txNodeId:
				continue
			dist_3d = calcDist(self.tx_node.x, rx_node.x, self.tx_node.y, rx_node.y, self.tx_node.z, rx_node.z) 
			offset = conf.LINK_OFFSET[(self.txNodeId, rx_node.nodeid)]
			self.LplAtN[rx_node.nodeid] = estimatePathLoss(dist_3d, self.freq, self.tx_node.z, rx_node.z) + offset
			self.rssiAtN[rx_node.nodeid] = self.txpow + self.tx_node.antennaGain + rx_node.antennaGain - self.LplAtN[rx_node.nodeid]
			if self.rssiAtN[rx_node.nodeid] >= conf.SENSMODEM[conf.MODEM]:
				self.sensedByN[rx_node.nodeid] = True
			if self.rssiAtN[rx_node.nodeid] >= conf.CADMODEM[conf.MODEM]:
				self.detectedByN[rx_node.nodeid] = True
				
		self.packetLen = plen
		self.timeOnAir = airtime(self.sf, self.cr, self.packetLen, self.bw)
		self.startTime = 0
		self.endTime = 0

		# Routing
		self.retransmissions = conf.maxRetransmission
		self.ackReceived = False
		self.hopLimit = self.tx_node.hopLimit

		self.previousCoverageFilter = coverageFilter
		self.totalNodesInCoverageFilter = prevNodesInCovFilter
		self.additionalCoverageRatio = 0.0
		self.neighbors = 0
		self.setCoverageFilter()

	def setCoverageFilter(self):
		self.coverageFilter = CoverageFilter()
		# Always add the transmitting node
		self.coverageFilter.add(self.txNodeId)

		# Merge prior coverage bits
		if self.previousCoverageFilter is not None:
			self.coverageFilter.merge(self.previousCoverageFilter)

		# This will prune coverage knowledge and return the latest
		coverageSet = self.tx_node.getCoverageKnowledge()
		for nodeid in coverageSet:
			self.coverageFilter.add(nodeid)
	
	def refreshAdditionalCoverageRatio(self):
		# If we don't have previous coverage, all coverage is new!
		if self.previousCoverageFilter is None:
			return 1

		newCoverage = 0
		newCoverageWeighted = 0
		numNodes = 0
		numNodesWeighted = 0
		for nodeid in self.tx_node.coverageKnowledge:
			# If this is the node transmitting the packet, or the original sender, skip
			# Simulates having `relay_node` in the header as well as `from`
			if nodeid == self.txNodeId or nodeid == self.origTxNodeId:
				continue

			lastHeard = self.tx_node.lastHeardTime[nodeid]
			if lastHeard is not None:
				# We have last heard time, so this is a node in OUR coverage
				# We don't yet know if its new coverage relative to the previous coverage
				numNodes += 1
				age = self.now - lastHeard
				if age < conf.RECENCY_THRESHOLD:
					recency = self.computeRecencyWeight(age, conf.RECENCY_THRESHOLD)
					# Add the "value" of this node to the total denominator
					numNodesWeighted += recency
					# If this node is not in the previous coverage, it's new coverage
					# Add the "value" of this node to the numerator
					if not self.previousCoverageFilter.check(nodeid):
						newCoverage += 1
						newCoverageWeighted += recency


		self.totalNodesInCoverageFilter += newCoverage
		self.neighbors = numNodes
		if numNodesWeighted > 0:
			self.additionalCoverageRatio = float(newCoverageWeighted) / float(numNodesWeighted)
		else:
			self.additionalCoverageRatio = 0.0
	
	def getRebroadcastProbability(self):
		self.refreshAdditionalCoverageRatio()

		rebroadcastProbability = conf.BASELINE_REBROADCAST_PROBABILITY + (self.additionalCoverageRatio * conf.COVERAGE_RATIO_SCALE_FACTOR)

		# Clamp to values that make sense
		rebroadcastProbability = max(0.0, min(1.0, rebroadcastProbability))

		return rebroadcastProbability
	
	def getPacketCoverageDifference(self, nodes):
		tx_id = self.txNodeId
		fp = 0  # false positives
		fn = 0  # false negatives

		for n in nodes:
			if n.nodeid == tx_id:
				continue  # skip the transmitter itself

			actuallyInRange = self.sensedByN[n.nodeid]  # True/False
			knowledgeSaysInRange = (n.nodeid in n.coverageKnowledge)

			# If physically not in range, but coverage knowledge says "in range"
			if (not actuallyInRange) and knowledgeSaysInRange:
				fp += 1
			# If physically in range, but coverage knowledge says "not in range"
			elif actuallyInRange and (not knowledgeSaysInRange):
				fn += 1

		return (fp, fn)

	def computeRecencyWeight(self, age, timeWindowSecs):
		"""
		age: (now - node.lastHeard) in seconds
		timeWindowSecs: The same recency threshold you use in firmware (e.g., 3600 if 1 hour).
		"""
		ratio = 1.0 - (float(age) / float(timeWindowSecs))
		# clamp to [0..1]
		return max(0.0, min(1.0, ratio))


	
class MeshMessage():
	def __init__(self, origTxNodeId, destId, genTime, seq):
		self.origTxNodeId = origTxNodeId
		self.destId = destId
		self.genTime = genTime
		self.seq = seq
		self.endTime = 0
