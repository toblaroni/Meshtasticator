from lib.common import calcDist
from lib.coverageFilter import CoverageFilter
from .phy import *


NODENUM_BROADCAST = 0xFFFFFFFF
random.seed(conf.SEED)


class MeshPacket(): 
	def __init__(self, nodes, origTxNodeId, destId, txNodeId, plen, seq, genTime, wantAck, isAck, requestId, coverageFilter = None, prevNodesInCovFilter = 0):
		self.origTxNodeId = origTxNodeId
		self.destId = destId
		self.txNodeId = txNodeId
		self.wantAck = wantAck
		self.isAck = isAck
		self.seq = seq
		self.requestId = requestId
		self.genTime = genTime
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
		self.setCoverageFilter()

	def setCoverageFilter(self):
		self.coverageFilter = CoverageFilter()
		# Always add the transmitting node
		self.coverageFilter.add(self.txNodeId)

		# Merge prior coverage bits
		if self.previousCoverageFilter is not None:
			self.coverageFilter.merge(self.previousCoverageFilter)

		coverageSet = self.tx_node.getCoverageKnowledge()
		for nodeid in coverageSet:
			self.coverageFilter.add(nodeid)
	
	def refreshAdditionalCoverageRatio(self):
		# If we don't have previous coverage, all coverage is new!
		if self.previousCoverageFilter is None:
			return 1

		newCoverage = 0
		numNodes = 0
		for nodeid, is_sensed in enumerate(self.sensedByN):
			if is_sensed:
				numNodes += 1
				if not self.previousCoverageFilter.check(nodeid):
					newCoverage += 1

		self.totalNodesInCoverageFilter += newCoverage
		if numNodes > 0:
			self.additionalCoverageRatio = float(newCoverage) / float(numNodes)
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
			knowledgeSaysInRange = (tx_id in n.coverageKnowledge)

			# If physically not in range, but coverage knowledge says "in range"
			if (not actuallyInRange) and knowledgeSaysInRange:
				fp += 1
			# If physically in range, but coverage knowledge says "not in range"
			elif actuallyInRange and (not knowledgeSaysInRange):
				fn += 1

		return (fp, fn)

	
class MeshMessage():
	def __init__(self, origTxNodeId, destId, genTime, seq):
		self.origTxNodeId = origTxNodeId
		self.destId = destId
		self.genTime = genTime
		self.seq = seq
		self.endTime = 0
