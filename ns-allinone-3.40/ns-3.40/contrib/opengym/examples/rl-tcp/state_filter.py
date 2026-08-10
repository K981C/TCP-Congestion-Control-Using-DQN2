import numpy as np
import torch

RAW_IDX = {
    "time_us":        2,
    "cwnd":           5,
    "segment_size":   6,
    "ssThresh":       4,          
    "segments_acked": 7,
    "bytes_in_flight":8,
    "rtt_us":         9,
    "min_rtt_us":     10,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Filter:
    def __init__(self):
        self.reset()

    def reset(self):
        self._start_time_us = None
        self._prev_time_us = None
        self._prev_cwnd = None
        self._prev_rtt_us = None
        self._total_lost = 0
        self.relative_time_s = 0
        
        # --- REMOVED: self._rtt_samples = [] ---
        self._rtt_count = 0            # <--- ADD THIS
        self._avg_rtt_running = 0.0    # <--- ADD THIS
        self._rtt_alpha = 0.05
        
        self._min_throughput = float('inf')
        self._max_throughput = float('-inf')
        self._min_rtt_ms = float('inf')
        self._max_rtt_ms = float('-inf')
        # running throughput estimator (Mbps)
        self._throughput_running = 0.0
        self._throughput_count = 0

    def extract(self, obs):
        now_us = float(obs[RAW_IDX["time_us"]])
        cwnd = float(obs[RAW_IDX["cwnd"]])
        seg_size = float(obs[RAW_IDX["segment_size"]]) or 1.0
        segs_acked = float(obs[RAW_IDX["segments_acked"]])
        bytes_flight = float(obs[RAW_IDX["bytes_in_flight"]])
        rtt_us = float(obs[RAW_IDX["rtt_us"]])
        min_rtt_us = float(obs[RAW_IDX["min_rtt_us"]]) or rtt_us
        ssThresh = float(obs[RAW_IDX["ssThresh"]])

        if self._start_time_us is None:
            self._start_time_us = now_us

        self.relative_time_s = (now_us - self._start_time_us) / 1e6

        cwnd_bytes = cwnd
        unacked_bytes = bytes_flight
        avg_rtt_ms = rtt_us / 1e3

        # Calculate instantaneous throughput (Mbps) guarded against tiny dt
        if self._prev_time_us is not None:
            dt_s = (now_us - self._prev_time_us) / 1e6
            min_dt = 1e-3
            dt_s = max(dt_s, min_dt)
            throughput_mbps = ((segs_acked * seg_size * 8) / (dt_s * 1e6)) if dt_s > 0 else 0.0
        else:
            throughput_mbps = 0.0

        # Smooth throughput with an EMA to remove spikes from tiny dt
        if self._throughput_count == 0:
            self._throughput_running = throughput_mbps
        else:
            alpha = 0.2
            self._throughput_running = self._throughput_running * (1.0 - alpha) + throughput_mbps * alpha
        self._throughput_count += 1
        throughput_mbps = self._throughput_running

        # Track packet losses. Treat obs[11] (calledFunc) as an event flag.
        # Return a per-step loss flag (0.0 or 1.0) for metrics/state so the
        # reward sees instantaneous loss events rather than a cumulative count.
        lost_flag = 1.0 if int(obs[11]) == 0 else 0.0

        # Keep a cumulative counter only for diagnostics (do not use in reward)
        if int(obs[11]) == 0:
            self._total_lost += 1

        # per-step loss indicator
        lost_packets = lost_flag

        # Calculate jitter
        jitter_ms = abs(rtt_us - self._prev_rtt_us) / 1e3 if self._prev_rtt_us is not None else 0.0

        # Calculate queue size estimate in bytes using RTT inflation
        if rtt_us > 0 and min_rtt_us > 0 and rtt_us > min_rtt_us:
            queue_size_bytes = bytes_flight * ((rtt_us - min_rtt_us) / rtt_us)
        else:
            queue_size_bytes = 0.0

        current_rtt_ms = rtt_us / 1e3

        # Update min/max trackers for metrics
        if self._prev_time_us is not None:
            self._min_throughput = min(self._min_throughput, throughput_mbps)
            self._max_throughput = max(self._max_throughput, throughput_mbps)

        self._min_rtt_ms = min(self._min_rtt_ms, current_rtt_ms)
        self._max_rtt_ms = max(self._max_rtt_ms, current_rtt_ms)

        self._prev_time_us = now_us
        self._prev_cwnd = cwnd
        self._prev_rtt_us = rtt_us

        # -----------------------------------------------------------
        # PERFORMANCE FIX: Running Average instead of infinite list
        # -----------------------------------------------------------
        if self._rtt_count == 0:
            self._avg_rtt_running = current_rtt_ms
        else:
            # Exponential Moving Average (faster than storing/mean-ing millions of items)
            self._avg_rtt_running = self._avg_rtt_running * (1.0 - self._rtt_alpha) + current_rtt_ms * self._rtt_alpha
        self._rtt_count += 1
        avg_rtt_running = self._avg_rtt_running

        # -----------------------------------------------------------
        # Absolute Scaling (kept from your fixed version)
        # -----------------------------------------------------------
        # Normalization constants (tune to your topology)
        CWND_SCALE = 1e6
        UNACKED_SCALE = 1e6
        SS_THRESH_SCALE = 1e6
        RTT_SCALE_MS = 1000.0
        TP_SCALE_MBPS = 2.0
        QUEUE_SCALE = 1e6

        raw_state = np.array([
            min(self.relative_time_s / 300.0, 1.0),
            cwnd_bytes / CWND_SCALE,
            unacked_bytes / UNACKED_SCALE,
            ssThresh / SS_THRESH_SCALE,
            avg_rtt_ms / RTT_SCALE_MS,
            min(throughput_mbps / TP_SCALE_MBPS, 1.0),
            lost_packets,
            jitter_ms / RTT_SCALE_MS,
            queue_size_bytes / QUEUE_SCALE,
        ], dtype=np.float32)

        state = np.clip(raw_state, 0.0, 1.0)

        # Metrics (returned raw, for the reward function)
        min_tp = self._min_throughput if self._min_throughput != float('inf') else 0.0
        max_tp = self._max_throughput if self._max_throughput != float('-inf') else 0.0
        min_rtt = self._min_rtt_ms if self._min_rtt_ms != float('inf') else 0.0
        max_rtt = self._max_rtt_ms if self._max_rtt_ms != float('-inf') else 0.0

        metrics = np.array([
            throughput_mbps,    # observed
            min_tp,             # min
            max_tp,             # max
            avg_rtt_running,    # avg (fast O(1) now!)
            min_rtt,            # min
            max_rtt,            # max
            lost_packets
            
        ], dtype=np.float32)

        return torch.from_numpy(state).to(device), metrics