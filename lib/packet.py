from lib.common import calcDist
from lib.coverageFilter import CoverageFilter
from .phy import *


NODENUM_BROADCAST = 0xFFFFFFFF
random.seed(conf.SEED)


class MeshPacket(): 
	def __init__(self, nodes, origTxNodeId, destId, txNodeId, plen, seq, genTime, wantAck, isAck, requestId, coverageFilter = None):
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
		tx_node = next(n for n in nodes if n.nodeid == self.txNodeId)
		for rx_node in nodes:
			if rx_node.nodeid == self.txNodeId:
				continue
			dist_3d = calcDist(tx_node.x, rx_node.x, tx_node.y, rx_node.y, tx_node.z, rx_node.z) 
			self.LplAtN[rx_node.nodeid] = estimatePathLoss(dist_3d, self.freq, tx_node.z, rx_node.z)
			self.rssiAtN[rx_node.nodeid] = self.txpow + tx_node.antennaGain + rx_node.antennaGain - self.LplAtN[rx_node.nodeid]
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
		self.hopLimit = tx_node.hopLimit

		self.previousCoverageFilter = coverageFilter;
		self.additionalCoverageRatio = 0.0;
		self.setCoverageFilter()

	def setCoverageFilter(self):
		# Always create a new, empty CoverageFilter
		self.coverageFilter = CoverageFilter()
		
		# Always add the transmitting node to the coverage filter
		self.coverageFilter.add(self.txNodeId)

		# If there was a previous coverage filter, merge its bits into the new one
		if self.previousCoverageFilter is not None:
			self.coverageFilter.merge(self.previousCoverageFilter)

		# Then add our own newly sensed coverage
		for nodeid, is_sensed in enumerate(self.sensedByN):
			if is_sensed:
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

		self.additionalCoverageRatio = float(newCoverage) / float(numNodes)
	
	def getRebroadcastProbability(self):
		self.refreshAdditionalCoverageRatio()

		rebroadcastProbability = conf.BASELINE_REBROADCAST_PROBABILITY + (self.additionalCoverageRatio * conf.COVERAGE_RATIO_SCALE_FACTOR)

		# Clamp to values that make sense
		rebroadcastProbability = max(0.0, min(1.0, rebroadcastProbability))

		return rebroadcastProbability
	
class MeshMessage():
	def __init__(self, origTxNodeId, destId, genTime, seq):
		self.origTxNodeId = origTxNodeId
		self.destId = destId
		self.genTime = genTime
		self.seq = seq
		self.endTime = 0
