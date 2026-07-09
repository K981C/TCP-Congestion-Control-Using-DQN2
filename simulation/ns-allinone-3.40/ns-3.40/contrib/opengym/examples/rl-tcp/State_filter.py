
import argparse
import numpy as np
from ns3gym import ns3env
import os
RAW_IDX = {
    "time_us":        2,
    "cwnd":           5,
    "segment_size":   6,
    "segments_acked": 7,
    "bytes_in_flight":8,
    "rtt_us":         9,
    "min_rtt_us":     10,
}

STATE_LABELS = [
    "relative_time_s",
    "cwnd_bytes",
    "unacked_bytes",
    "ack_packets",
    "avg_rtt_ms",
    "throughput_mbps",
    "lost_packets",
    "jitter_ms",
    "queue_size_bytes",
]

METRICS_LABELS = [
    "observed_throughput_mbps",  
    "min_throughput_mbps",        
    "max_throughput_mbps",       
    "avg_rtt_ms",               
    "min_rtt_ms",             
    "max_rtt_ms",                
]


class StateExtractor:
    def __init__(self):
        self.reset()

    def reset(self):
        self._start_time_us  = None
        self._prev_time_us   = None
        self._prev_cwnd      = None
        self._prev_rtt_us    = None
        self._prev_acked     = 0
        self._total_lost     = 0
        self.relative_time_s = 0
       
        self._rtt_samples        = [] 
        self._min_throughput     = float('inf')
        self._max_throughput     = float('-inf')
        self._min_rtt_ms         = float('inf')
        self._max_rtt_ms         = float('-inf')

        self._state_min = np.array([
            0.0,    # relative_time_s
            0.0,    # cwnd_bytes
            0.0,    # unacked_bytes
            0.0,    # ack_packets
            0.0,    # avg_rtt_ms
            0.0,    # throughput_mbps
            0.0,    # lost_packets
            0.0,    # jitter_ms
            0.0,    # queue_size_bytes
        ], dtype=np.float32)

        self._state_max = np.array([
            1.0,      # relative_time_s
            200000.0, # cwnd_bytes
            200000.0, # unacked_bytes
            100.0,    # ack_packets
            1000.0,   # avg_rtt_ms
            1000.0,   # throughput_mbps
            1000.0,   # lost_packets
            1000.0,   # jitter_ms
            200000.0, # queue_size_bytes
        ], dtype=np.float32)

    def extract(self, obs):

        now_us        = float(obs[RAW_IDX["time_us"]])
        cwnd          = float(obs[RAW_IDX["cwnd"]])
        seg_size      = float(obs[RAW_IDX["segment_size"]]) or 1.0
        segs_acked    = float(obs[RAW_IDX["segments_acked"]])
        bytes_flight  = float(obs[RAW_IDX["bytes_in_flight"]])
        rtt_us        = float(obs[RAW_IDX["rtt_us"]])
        min_rtt_us    = float(obs[RAW_IDX["min_rtt_us"]]) or rtt_us

        if self._start_time_us is None:
            self._start_time_us = now_us

      
        self.relative_time_s  =(now_us - self._start_time_us) / 1e6
        self._start_time_us = now_us
        cwnd_bytes       = cwnd
        unacked_bytes    = bytes_flight
        ack_packets      = segs_acked
        avg_rtt_ms       = rtt_us / 1e3

        
        if self._prev_time_us is not None:
            dt_s = (now_us - self._prev_time_us) / 1e6
            throughput_mbps = ((segs_acked * seg_size * 8) / (dt_s * 1e6)) if dt_s > 0 else 0.0
        else:
            throughput_mbps = 0.0

        if self._prev_cwnd is not None and cwnd < self._prev_cwnd:
            self._total_lost += max(0.0, (self._prev_cwnd - cwnd) / seg_size)
        lost_packets = self._total_lost

        jitter_ms = abs(rtt_us - self._prev_rtt_us) / 1e3 if self._prev_rtt_us is not None else 0.0

        if rtt_us > 0:
            queue_size_bytes = max(0.0, bytes_flight - cwnd * (min_rtt_us / rtt_us))
        else:
            queue_size_bytes = 0.0

       
        current_rtt_ms = rtt_us / 1e3
        self._rtt_samples.append(current_rtt_ms)

        
        if self._prev_time_us is not None:
            self._min_throughput = min(self._min_throughput, throughput_mbps)
            self._max_throughput = max(self._max_throughput, throughput_mbps)

        self._min_rtt_ms = min(self._min_rtt_ms, current_rtt_ms)
        self._max_rtt_ms = max(self._max_rtt_ms, current_rtt_ms)

      
        self._prev_time_us = now_us
        self._prev_cwnd    = cwnd
        self._prev_rtt_us  = rtt_us
        self._prev_acked   = segs_acked

       
        state = np.array([
            self.relative_time_s,
            cwnd_bytes ,
            unacked_bytes,
            ack_packets,
            avg_rtt_ms,
            throughput_mbps,
            lost_packets,
            jitter_ms,
            queue_size_bytes,
        ], dtype=np.float32)

        # Min-max normalize state to [0, 1]
        denom = self._state_max - self._state_min
        denom[denom == 0.0] = 1.0
        state = np.clip((state - self._state_min) / denom, 0.0, 1.0)

        min_tp  = self._min_throughput if self._min_throughput != float('inf')  else 0.0
        max_tp  = self._max_throughput if self._max_throughput != float('-inf') else 0.0
        min_rtt = self._min_rtt_ms     if self._min_rtt_ms     != float('inf')  else 0.0
        max_rtt = self._max_rtt_ms     if self._max_rtt_ms     != float('-inf') else 0.0
        avg_rtt_running = float(np.mean(self._rtt_samples)) if self._rtt_samples else 0.0

        metrics = np.array([
            throughput_mbps,    # 0  observed_throughput_mbps  (this step)
            min_tp,             # 1  min_throughput_mbps       (running min)
            max_tp,             # 2  max_throughput_mbps       (running max)
            avg_rtt_running,    # 3  avg_rtt_ms                (running average)
            min_rtt,            # 4  min_rtt_ms                (running min)
            max_rtt,            # 5  max_rtt_ms                (running max)
        ], dtype=np.float32)

        return state, metrics

parser = argparse.ArgumentParser()
parser.add_argument('--start',      type=int, default=1)
parser.add_argument('--iterations', type=int, default=1)
args = parser.parse_args()

env       = ns3env.Ns3Env(port=5555, startSim=bool(args.start),
                          simSeed=12, simArgs={"--duration": 36000}, debug=False)

extractor = StateExtractor()
def Agent(action):
    try:
      
        
        obs, reward, done, info = env.step(action)

        if(done):
            env.reset()
            extractor.reset()
            env.close()
        else:
            state, metrics = extractor.extract(obs)
            return state, metrics

    


    finally:
        """finallly"""
        # env.reset()
        # extractor.reset()
        # env.close()
