#!/usr/bin/env python3
import sys

from lib.common import *
from lib.discrete_event import *
from lib.mac import *
from lib.packet import *


class MeshNode():
    def __init__(self, conf, nodes, env, bc_pipe, nodeid, period, messages, packetsAtN, packets, delays, nodeConfig, messageSeq, verboseprint):
        self.conf = conf
        self.nodeid = nodeid
        self.verboseprint = verboseprint
        self.moveRng = random.Random(nodeid)
        self.nodeRng = random.Random(nodeid)
        self.rebroadcastRng = random.Random()
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
            self.x, self.y = findRandomPosition(self.conf, nodes)
            self.z = self.conf.HM
            self.isRouter = self.conf.router
            self.isRepeater = False
            self.isClientMute = False
            self.hopLimit = self.conf.hopLimit
            self.antennaGain = self.conf.GL
        # messageSeq: Dict { 'val': <num> }
        # Everytime a packet gets sent the messageSeq of the node gets incremented
        # This Dict is **shared** across all nodes so that messages have unique IDs...
        self.messageSeq = messageSeq    
        self.env = env
        self.period = period
        self.bc_pipe = bc_pipe
        self.rx_snr = 0
        self.nodes = nodes
        self.messages = messages    # Messages **this** node has sent
        self.packetsAtN = packetsAtN
        self.nrPacketsSent = 0
        self.packets = packets
        self.delays = delays
        self.leastReceivedHopLimit = {} # Keeps track of requests (for this node) with the lowest hop (for ACKs)
        self.isReceiving = []
        self.isTransmitting = False
        self.usefulPackets = 0  # What is this..?
        self.txAirUtilization = 0
        self.airUtilization = 0
        self.droppedByDelay = 0
        self.rebroadcastPackets = 0
        self.isMoving = False
        self.gpsEnabled = False
        # Track last broadcast position/time
        self.lastBroadcastX = self.x
        self.lastBroadcastY = self.y
        self.lastBroadcastTime = 0
        # track total transmit time for the last 6 buckets (each is 10s in firmware logic)
        self.channelUtilization = [0]*self.conf.CHANNEL_UTILIZATION_PERIODS  # each entry is ms spent on air in that interval
        self.channelUtilizationIndex = 0  # which "bucket" is current
        self.prevTxAirUtilization = 0.0   # how much total tx air-time had been used at last sample

        # GOSSIP
        self.receivedImplAck = {}   # Dictionary of this nodes packets and whether we've received an implicit ACK 
        self.seenPackets = set()    # Packets received from other nodes. This is useful if we want to implement ACKs and Retransmissions for GOSSIP

        env.process(self.trackChannelUtilization(env))
        if not self.isRepeater:  # repeaters don't generate messages themselves
            env.process(self.generateMessage())
        env.process(self.receive(self.bc_pipe.get_output_conn()))
       # One transmitter per node (essentially a lock). Only one process can use
        self.transmitter = simpy.Resource(env, 1)

        # start mobility if enabled
        if self.conf.MOVEMENT_ENABLED and self.moveRng.random() <= self.conf.APPROX_RATIO_NODES_MOVING:
            self.isMoving = True
            if self.moveRng.random() <= self.conf.APPROX_RATIO_OF_NODES_MOVING_W_GPS_ENABLED:
                self.gpsEnabled = True

            # Randomly assign a movement speed
            possibleSpeeds = [
                self.conf.WALKING_METERS_PER_MIN,  # e.g.,  96 m/min
                self.conf.BIKING_METERS_PER_MIN,   # e.g., 390 m/min
                self.conf.DRIVING_METERS_PER_MIN   # e.g., 1500 m/min
            ]
            self.movementStepSize = self.moveRng.choice(possibleSpeeds)

            env.process(self.moveNode(env))

    def trackChannelUtilization(self, env):
        """
        Periodically compute how many seconds of airtime this node consumed
        over the last 10-second block and store it in the ring buffer.
        """
        while True:
            # Wait 10 seconds of simulated time
            yield env.timeout(self.conf.TEN_SECONDS_INTERVAL)

            curTotalAirtime = self.txAirUtilization  # total so far, in *milliseconds*
            blockAirtimeMs = curTotalAirtime - self.prevTxAirUtilization

            self.channelUtilization[self.channelUtilizationIndex] = blockAirtimeMs

            self.prevTxAirUtilization = curTotalAirtime
            self.channelUtilizationIndex = (self.channelUtilizationIndex + 1) % self.conf.CHANNEL_UTILIZATION_PERIODS

    def channelUtilizationPercent(self) -> float:
        """
        Returns how much of the last 60 seconds (6 x 10s) this node spent transmitting, as a percent.
        """
        sumMs = sum(self.channelUtilization)
        # 6 intervals, each 10 seconds = 60,000 ms total
        # fraction = sum_ms / 60000, then multiply by 100 for percent
        return (sumMs / (self.conf.CHANNEL_UTILIZATION_PERIODS * self.conf.TEN_SECONDS_INTERVAL)) * 100.0

    def moveNode(self, env):
        while True:

            # Pick a random direction and distance
            angle = 2 * math.pi * self.moveRng.random()
            distance = self.movementStepSize * self.moveRng.random()
            
            # Compute new position
            dx = distance * math.cos(angle)
            dy = distance * math.sin(angle)
            
            leftBound   = self.conf.OX - self.conf.XSIZE/2
            rightBound  = self.conf.OX + self.conf.XSIZE/2
            bottomBound = self.conf.OY - self.conf.YSIZE/2
            topBound    = self.conf.OY + self.conf.YSIZE/2

            # Then in moveNode:
            new_x = min(max(self.x + dx, leftBound), rightBound)
            new_y = min(max(self.y + dy, bottomBound), topBound)
            
            # Update node’s position
            self.x = new_x
            self.y = new_y

            if self.gpsEnabled:
                distanceTraveled = calcDist(self.lastBroadcastX, self.x, self.lastBroadcastY, self.y)
                timeElapsed = env.now - self.lastBroadcastTime
                if (distanceTraveled >= self.conf.SMART_POSITION_DISTANCE_THRESHOLD and
                    timeElapsed >= self.conf.SMART_POSITION_DISTANCE_MIN_TIME):

                    currentUtil = self.channelUtilizationPercent()
                    if currentUtil < 25.0:
                        self.sendPacket(NODENUM_BROADCAST, "POSITION")
                        self.lastBroadcastX = self.x
                        self.lastBroadcastY = self.y
                        self.lastBroadcastTime = env.now
                    else:
                        self.verboseprint(f"At time {env.now} node {self.nodeid} SKIPS POSITION broadcast (util={currentUtil:.1f}% > 25%)")

            
            # Wait until next move
            nextMove = self.getNextTime(self.conf.ONE_MIN_INTERVAL)
            if nextMove >= 0:
                yield env.timeout(nextMove)
            else:
                break

    def sendPacket(self, destId, type=""):
        # increment the shared counter
        self.messageSeq["val"] += 1
        messageSeq = self.messageSeq["val"]
        self.messages.append(MeshMessage(self.nodeid, destId, self.env.now, messageSeq))
        p = None
        if self.conf.SELECTED_ROUTER_TYPE == self.conf.ROUTER_TYPE.GOSSIP:
            p = MeshPacket(self.conf, self.nodes, self.nodeid, destId, self.nodeid, self.conf.PACKETLENGTH, messageSeq, self.env.now, False, False, None, self.env.now, self.verboseprint)
            self.receivedImplAck[messageSeq] = False
        else:
            p = MeshPacket(self.conf, self.nodes, self.nodeid, destId, self.nodeid, self.conf.PACKETLENGTH, messageSeq, self.env.now, True, False, None, self.env.now, self.verboseprint)
        self.packets.append(p)
        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'generated', type, 'message', p.seq, 'to', destId)
        self.env.process(self.transmit(p))
        return p

    def getNextTime(self, period):  
        # Generates time until next message should be generated
        nextGen = self.nodeRng.expovariate(1.0/float(period))
        # do not generate message near the end of the simulation (otherwise flooding cannot finish in time)
        # `self.hopLimit or self.conf.initalHops` => for GOSSIP packets get propagated with p=1 for initialHops
        if self.env.now+nextGen+(self.hopLimit or self.conf.GOSSIP_K)*airtime(self.conf, self.conf.SFMODEM[self.conf.MODEM], self.conf.CRMODEM[self.conf.MODEM], self.conf.PACKETLENGTH, self.conf.BWMODEM[self.conf.MODEM]) < self.conf.SIMTIME:
            return nextGen
        return -1

    def generateMessage(self):
        while True:
            # Returns -1 if we won't make it before the sim ends
            nextGen = self.getNextTime(self.period)
            # do not generate message near the end of the simulation (otherwise flooding cannot finish in time)
            if nextGen >= 0:
                yield self.env.timeout(nextGen) 

                if self.conf.DMs:
                    destId = self.nodeRng.choice([i for i in range(0, len(self.nodes)) if i is not self.nodeid])
                else:
                    destId = NODENUM_BROADCAST

                p = self.sendPacket(destId)
                
                while p.wantAck: # ReliableRouter: retransmit message if no ACK received after timeout 
                    retransmissionMsec = getRetransmissionMsec(self, p) 
                    yield self.env.timeout(retransmissionMsec)

                    ackReceived = False  # check whether you received an ACK on the transmitted message
                    minRetransmissions = self.conf.maxRetransmission
                    for packetSent in self.packets:
                        if packetSent.origTxNodeId == self.nodeid and packetSent.seq == p.seq:
                            if packetSent.retransmissions < minRetransmissions:
                                minRetransmissions = packetSent.retransmissions
                            if packetSent.ackReceived:
                                ackReceived = True
                    if ackReceived: 
                        self.verboseprint('Node', self.nodeid, 'received ACK on generated message with seq. nr.', p.seq)
                        break
                    else: 
                        if minRetransmissions > 0:  # generate new packet with same sequence number
                            pNew = MeshPacket(self.conf, self.nodes, self.nodeid, p.destId, self.nodeid, p.packetLen, p.seq, p.genTime, p.wantAck, False, None, self.env.now, self.verboseprint)
                            pNew.retransmissions = minRetransmissions-1
                            self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'wants to retransmit its generated packet to', destId, 'with seq.nr.', p.seq, 'minRetransmissions', minRetransmissions)
                            self.packets.append(pNew)
                            self.env.process(self.transmit(pNew))
                        else:
                            self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'reliable send of', p.seq, 'failed.')
                            break
            else:  # do not send this message anymore, since it is close to the end of the simulation
                break


    def transmit(self, packet):
        with self.transmitter.request() as request:
            yield request

            # listen-before-talk from src/mesh/RadioLibInterface.cpp 
            if self.conf.SELECTED_ROUTER_TYPE != self.conf.ROUTER_TYPE.GOSSIP:
                txTime = setTransmitDelay(self, packet) 
                self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'picked wait time', txTime)
                yield self.env.timeout(txTime)
                self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'ends waiting')

            # wait when currently receiving or transmitting, or channel is active
            while any(self.isReceiving) or self.isTransmitting or isChannelActive(self, self.env):
                self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'is busy Tx-ing', self.isTransmitting, 'or Rx-ing', any(self.isReceiving), 'else channel busy!')
                txTime = setTransmitDelay(self, packet)     # txTime = back off time
                yield self.env.timeout(txTime)
                self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'ends waiting')

            if self.conf.SELECTED_ROUTER_TYPE == self.conf.ROUTER_TYPE.GOSSIP:  # GOSSIP
                # This is mainly to prevent unnecessary retransmissions
                # if self.receivedImplAck.get(packet.seq) == True:
                if packet.ackReceived:
                    self.verboseprint('(GOSSIP) At time', round(self.env.now, 3), 'node', self.nodeid, 'in the meantime received ACK, abort packet with seq. nr', packet.seq)
                    self.packets.remove(packet)
                    return
            else:   # Non-GOSSIP routing
                # check if you received an ACK for this message in the meantime
                if packet.seq not in self.leastReceivedHopLimit:
                    self.leastReceivedHopLimit[packet.seq] = packet.hopLimit+1 

                if not packet.ackReceived:
                    # no ACK received yet, so may start transmitting 
                    self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'started low level send', packet.seq, 'hopLimit', packet.hopLimit, 'original Tx', packet.origTxNodeId)
                else:  # received ACK: abort transmit, remove from packets generated 
                    self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'in the meantime received ACK, abort packet with seq. nr', packet.seq)
                    self.packets.remove(packet)
                    return

            # Transmit
            self.nrPacketsSent += 1
            for rx_node in self.nodes:
                if packet.sensedByN[rx_node.nodeid] == True:
                    if (checkcollision(self.conf, self.env, packet, rx_node.nodeid, self.packetsAtN) == 0):
                        self.packetsAtN[rx_node.nodeid].append(packet)
            packet.startTime = self.env.now
            packet.endTime = self.env.now + packet.timeOnAir
            self.txAirUtilization += packet.timeOnAir
            self.airUtilization += packet.timeOnAir

            self.bc_pipe.put(packet)
            self.isTransmitting = True
            yield self.env.timeout(packet.timeOnAir)
            self.isTransmitting = False


    def receive(self, in_pipe):
        while True:
            # Wait for packet to become available in Simpy pipeline
            p = yield in_pipe.get()
            if p.sensedByN[self.nodeid] and not p.collidedAtN[self.nodeid] and p.onAirToN[self.nodeid]:  # start of reception
                if not self.isTransmitting:
                    self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'started receiving packet', p.seq, 'from', p.txNodeId)
                    p.onAirToN[self.nodeid] = False 
                    self.isReceiving.append(True)
                else:  # if you were currently transmitting, you could not have sensed it
                    self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'was transmitting, so could not receive packet', p.seq)
                    p.sensedByN[self.nodeid] = False
                    p.onAirToN[self.nodeid] = False
            elif p.sensedByN[self.nodeid]:  # end of reception
                try: 
                    self.isReceiving[self.isReceiving.index(True)] = False 
                except: 
                    pass
                self.airUtilization += p.timeOnAir
                if p.collidedAtN[self.nodeid]:
                    self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'could not decode packet.')
                    continue
                p.receivedAtN[self.nodeid] = True
                self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'received packet', p.seq, 'with delay', round(self.env.now-p.genTime, 2))
                self.delays.append(self.env.now-p.genTime)

                if self.conf.SELECTED_ROUTER_TYPE == self.conf.ROUTER_TYPE.GOSSIP:  # GOSSIP
                    if p.seq not in self.seenPackets:
                        self.usefulPackets += 1
                        self.seenPackets.add(p.seq)
                    else:
                        self.verboseprint('====== Already seen this packet =======')
                        self.verboseprint('====== NOT USEFUL =======')
                else: # Non GOSSIP
                    # update hopLimit for this message
                    if p.seq not in self.leastReceivedHopLimit:  # did not yet receive packet with this seq nr.
                        # self.verboseprint('Node', self.nodeid, 'received packet nr.', p.seq, 'orig. Tx', p.origTxNodeId, "for the first time.")
                        self.usefulPackets += 1
                        self.leastReceivedHopLimit[p.seq] = p.hopLimit
                    if p.hopLimit < self.leastReceivedHopLimit[p.seq]:  # hop limit of received packet is lower than previously received one
                        self.leastReceivedHopLimit[p.seq] = p.hopLimit

                # check if implicit ACK for own generated message
                if p.origTxNodeId == self.nodeid:
                    if p.isAck:
                        self.verboseprint('Node', self.nodeid, 'received real ACK on generated message.')
                    else:
                        self.verboseprint('Node', self.nodeid, 'received implicit ACK on message sent.')
                    p.ackReceived = True

                    if self.conf.SELECTED_ROUTER_TYPE == self.conf.ROUTER_TYPE.GOSSIP:
                        self.receivedImplAck[p.seq] = True
                    continue

                ackReceived = False
                realAckReceived = False
                for sentPacket in self.packets:
                    # check if ACK for message you currently have in queue
                    if sentPacket.txNodeId == self.nodeid and sentPacket.seq == p.seq:
                        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'received implicit ACK for message in queue.')
                        ackReceived = True
                        sentPacket.ackReceived = True
                        if self.conf.SELECTED_ROUTER_TYPE == self.conf.ROUTER_TYPE.GOSSIP:
                            self.receivedImplAck[sentPacket.seq] = True
                    # check if real ACK for message sent
                    if sentPacket.origTxNodeId == self.nodeid and p.isAck and sentPacket.seq == p.requestId:
                        self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'received real ACK.')
                        realAckReceived = True
                        sentPacket.ackReceived = True

                # send real ACK if you are the destination and you did not yet send the ACK
                if p.wantAck and p.destId == self.nodeid and not any(pA.requestId == p.seq for pA in self.packets):
                    self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'sends a flooding ACK.')
                    self.messageSeq["val"] += 1
                    messageSeq = self.messageSeq["val"]
                    self.messages.append(MeshMessage(self.nodeid, p.origTxNodeId, self.env.now, messageSeq))
                    pAck = MeshPacket(self.conf, self.nodes, self.nodeid, p.origTxNodeId, self.nodeid, self.conf.ACKLENGTH, messageSeq, self.env.now, False, True, p.seq, self.env.now, self.verboseprint) 
                    self.packets.append(pAck)
                    self.env.process(self.transmit(pAck))
                # Rebroadcasting Logic for received message. This is a broadcast or a DM not meant for us.
                elif self.conf.SELECTED_ROUTER_TYPE == self.conf.ROUTER_TYPE.MANAGED_FLOOD:
                    if not p.destId == self.nodeid and not ackReceived and not realAckReceived and p.hopLimit > 0:
                    # FloodingRouter: rebroadcast received packet
                        if not self.isClientMute:
                            self.verboseprint('At time', round(self.env.now, 3), 'node', self.nodeid, 'rebroadcasts received packet', p.seq)
                            pNew = MeshPacket(self.conf, self.nodes, p.origTxNodeId, p.destId, self.nodeid, p.packetLen, p.seq, p.genTime, p.wantAck, False, None, self.env.now, self.verboseprint) 
                            pNew.hopLimit = p.hopLimit-1
                            self.packets.append(pNew)
                            self.env.process(self.transmit(pNew))
                elif self.conf.SELECTED_ROUTER_TYPE == self.conf.ROUTER_TYPE.GOSSIP:
                    if not self.isClientMute:
                        # In GOSSIP routing we will just rebroadcast with probability 1 for k hops and p after this
                        # Even if this node sent it originally we rebroadcast
                        if p.hopCount+1 >= self.conf.GOSSIP_K: 
                            if random.uniform(0, 1) < self.conf.GOSSIP_P:
                                self.verboseprint('(GOSSIP) At time', round(self.env.now, 3), 'node', self.nodeid, 'rebroadcasts received packet', p.seq, 'with probability', self.conf.GOSSIP_P)
                                pNew = MeshPacket(self.conf, self.nodes, p.origTxNodeId, p.destId, self.nodeid, p.packetLen, p.seq, p.genTime, p.wantAck, False, None, self.env.now, self.verboseprint) 
                                pNew.hopCount = p.hopCount+1
                                self.packets.append(pNew)
                                self.env.process(self.transmit(pNew))
                            else:
                                self.verboseprint('(GOSSIP) At time', round(self.env.now, 3), 'node', self.nodeid, 'abandons rebroadcast.')
                        else:
                            self.verboseprint('(GOSSIP) At time', round(self.env.now, 3), 'node', self.nodeid, 'rebroadcasts received packet', p.seq, 'with probability 1')
                            pNew = MeshPacket(self.conf, self.nodes, p.origTxNodeId, p.destId, self.nodeid, p.packetLen, p.seq, p.genTime, p.wantAck, False, None, self.env.now, self.verboseprint) 
                            pNew.hopCount = p.hopCount+1
                            self.packets.append(pNew)
                            self.env.process(self.transmit(pNew))
                else:
                    self.droppedByDelay += 1
