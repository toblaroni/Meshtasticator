from enum import Enum
import numpy as np


MODEL = 5  # Pathloss model to use (see README)

XSIZE = 15000  # horizontal size of the area to simulate in m 
YSIZE = 15000  # vertical size of the area to simulate in m
OX = 0.0  # origin x-coordinate
OY = 0.0  # origin y-coordinate
MINDIST = 15  # minimum distance between each node in the area in m

GL = 0  # antenna gain of each node in dBi
HM = 1.0  # height of each node in m

### Meshtastic specific ###
hopLimit = 3  # default 3
router = False  # set role of each node as router (True) or normal client (False)
maxRetransmission = 3  # default 3 -- not configurable by Meshtastic
### End of Meshtastic specific ###

ONE_SECOND_INTERVAL = 1000
TEN_SECONDS_INTERVAL = ONE_SECOND_INTERVAL * 10
ONE_MIN_INTERVAL = TEN_SECONDS_INTERVAL * 6
ONE_HR_INTERVAL = ONE_MIN_INTERVAL * 60

### Discrete-event specific ###
MODEM = 4  # LoRa modem to use: 0 = ShortFast, 1 = Short Slow, ... 7 = Very Long Slow (default 4 is LongFast)
PERIOD = 100 * ONE_SECOND_INTERVAL  # mean period of generating a new message with exponential distribution in ms
PACKETLENGTH = 40  # payload in bytes  
SIMTIME = 120 * ONE_MIN_INTERVAL  # duration of one simulation in ms
INTERFERENCE_LEVEL = 0.05  # chance that at a given moment there is already a LoRa packet being sent on your channel, 
                           # outside of the Meshtastic traffic. Given in a ratio from 0 to 1.  
COLLISION_DUE_TO_INTERFERENCE = False
DMs = False  # Set True for sending DMs (with random destination), False for broadcasts
# from RadioInterface.cpp RegionInfo regions[]
regions = { "US": {"freq_start": 902e6, "freq_end": 928e6, "power_limit": 30},
            "EU433": {"freq_start": 433e6, "freq_end": 434e6, "power_limit": 12}, 
            "EU868": {"freq_start": 868e6, "freq_end": 868e6, "power_limit": 27}}
REGION = regions["US"] # Select a different region here
CHANNEL_NUM = 27  # Channel number 
### End of discrete-event specific ###

### PHY parameters (normally no change needed) ###
PTX = REGION["power_limit"]
# from RadioInterface::applyModemConfig() 
BWMODEM = np.array([250e3, 250e3, 250e3, 250e3, 250e3, 125e3, 125e3, 62.5e3])  # bandwidth
SFMODEM = np.array([7, 8, 9, 10, 11, 11, 12, 12]) # spreading factor
CRMODEM = np.array([8, 8, 8, 8, 8, 8, 8, 8]) # coding rate
# minimum sensitivity from https://www.rfwireless-world.com/calculators/LoRa-Sensitivity-Calculator.html 
SENSMODEM = np.array([-121.5, -124.0, -126.5, -129.0, -131.5, -134.5, -137.0, -140.0])
# minimum received power for CAD (3dB less than sensitivity)
CADMODEM = np.array([-124.5, -127.0, -129.5, -132.0, -134.5, -137.5, -140.0, -143.0])
FREQ = REGION["freq_start"]+BWMODEM[MODEM]*CHANNEL_NUM
HEADERLENGTH = 16  # number of Meshtastic header bytes 
ACKLENGTH = 2  # ACK payload in bytes
NOISE_LEVEL = -119.25  # some noise level in dB, based on SNR_MIN and minimum receiver sensitivity
GAMMA = 2.08  # PHY parameter
D0 = 40.0  # PHY parameter
LPLD0 = 127.41  # PHY parameter
NPREAM = 16   # number of preamble symbols from RadioInterface.h 
### End of PHY parameters ###

# Misc
SEED = 44  # random seed to use
PLOT = True
RANDOM = False
# End of misc

# Initializers
NR_NODES = None
# End of initializers 

class ROUTER_TYPE(Enum):
    MANAGED_FLOOD = 'MANAGED_FLOOD'
    BLOOM = 'BLOOM'

######################################
####### SET ROUTER TYPE BELOW ########
######################################

SELECTED_ROUTER_TYPE = ROUTER_TYPE.BLOOM

######################################
####### SET ROUTER TYPE ABOVE ########
######################################

def updateRouterDependencies():
    # Overwrite hop limit in the case of Bloom routing
    if SELECTED_ROUTER_TYPE == ROUTER_TYPE.BLOOM:
        hopLimit = 15
        HEADERLENGTH = 32

updateRouterDependencies()

# Shrink this down to accomodate a 4 byte node id for relay_node
BLOOM_FILTER_SIZE_BYTES = 13
BLOOM_FILTER_SIZE_BITS = BLOOM_FILTER_SIZE_BYTES * 8

# This will scale up the impact of the coverage 
# ratio on probability of rebroadcast
COVERAGE_RATIO_SCALE_FACTOR = 3.0

# Set this to non-zero value to make it possible that a 
# node without any additional coverage may still rebroadcast
BASELINE_REBROADCAST_PROBABILITY = 0.0
UNKNOWN_COVERAGE_REBROADCAST_PROBABILITY = 0.8

SHOW_PROBABILITY_FUNCTION_COMPARISON = False

MAX_NEIGHBORS_PER_HOP = 20

# If SIMTIME is less than this, nodes will never fully age out of coverage
RECENCY_THRESHOLD = 4 * ONE_HR_INTERVAL

# Set this to True to enable the asymmetric link model
# Adds a random offset to the link quality of each link
MODEL_ASYMMETRIC_LINKS = True
MODEL_ASYMMETRIC_LINKS_MEAN = 0
MODEL_ASYMMETRIC_LINKS_STDDEV = 4
# Stores the offset for each link
# Populated when the simulator first starts
LINK_OFFSET = {}

MOVEMENT_ENABLED = True
WALKING_METERS_PER_MIN = 96
BIKING_METERS_PER_MIN = 390
DRIVING_METERS_PER_MIN = 1500

SMART_POSITION_DISTANCE_THRESHOLD = 100
# 30s minimum time in firmware
SMART_POSITION_DISTANCE_MIN_TIME = 30 * ONE_SECOND_INTERVAL

APPROX_RATIO_NODES_MOVING = 0.4

CHANNEL_UTILIZATION_PERIODS = 6