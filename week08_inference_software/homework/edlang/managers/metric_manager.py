import time
from collections import deque
import torch

METRIC_SHOW_PERIOD = 3.0

class MetricManager:
    def __init__(self, enable_metrics: bool = False):

        self.enable_metrics = enable_metrics
        self.waiting_queue_num = 0
        self.active_requests_num = 0

        self.throughput_tokens_per_second = 0.0
        self.ttft_ms = 0.0
        self.ttft_count = 0
        self.tpot_ms = 0.0
        self.tpot_count = 0
        self.rps = 0.0
        self.time = time.time()
        self.idx2time = {}
        self.decoding_requests_num = 0
        self.decoding_start_time = None
        self._pending_decode_events = deque()
        self._current_start_event = None
        self.avg_len = 0
        self.avg_len_count = 0

    def add_requests_prefill(self, req_ids):
        for req_id in req_ids:
            self.idx2time[req_id] = time.time()

    def remove_requests_prefill(self, req_ids):
        for req_id in req_ids:
            was_time = self.idx2time.pop(req_id)
            if self.ttft_count == 0:
                self.ttft_ms = (time.time() - was_time) * 1000
                self.ttft_count = 1
            else:
                self.ttft_ms = (self.ttft_ms * self.ttft_count + (time.time() - was_time) * 1000) / (self.ttft_count + 1)
                self.ttft_count += 1

    def update_lens(self, lens):
        if not lens:
            return
        if self.avg_len_count == 0:
            self.avg_len = sum(lens) / len(lens)
            self.avg_len_count = len(lens)
        else:
            self.avg_len = (self.avg_len * self.avg_len_count + sum(lens)) / (self.avg_len_count + len(lens))
            self.avg_len_count += len(lens)

    def calculate_rps(self):
        if self.throughput_tokens_per_second == 0:
            return 0.0
        per_request = self.ttft_ms / 1000 + self.avg_len / self.throughput_tokens_per_second
        self.rps = 1.0 / per_request
        return self.rps

    def calculate_throughtput_tokens_per_second(self):
        # Уже считается в flush_tpot
        # Делаю синк для первого запуска, чтобы получить представление о скорости, дальше использую только записаные события
        if self.tpot_count == 0:
            torch.cuda.synchronize()
            self.flush_tpot()

    def started_decoding(self):
        self.decoding_start_time = time.time()
        self._current_start_event = torch.cuda.Event(enable_timing=True)
        self._current_start_event.record()

    def finished_decoding(self, requests_num: int):
        if not requests_num:
            return
        self.decoding_requests_num = requests_num
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()
        self._pending_decode_events.append((self._current_start_event, end_event, requests_num))

    def flush_tpot(self):
        while self._pending_decode_events:
            start_ev, end_ev, num = self._pending_decode_events[0]
            if not end_ev.query():
                break
            self._pending_decode_events.popleft()
            elapsed_ms = start_ev.elapsed_time(end_ev)
            if self.tpot_count == 0:
                self.tpot_ms = elapsed_ms
                self.tpot_count = num
            else:
                self.tpot_ms = (self.tpot_ms * self.tpot_count + elapsed_ms * num) / (self.tpot_count + num)
                self.tpot_count += num
            self.throughput_tokens_per_second = num / (elapsed_ms / 1000)

    def update_waiting_queue_num(self, num: int):
        self.waiting_queue_num = num

    def update_active_requests_num(self, num: int):
        self.active_requests_num = num

    def show_metrics(self, stage: str):
        metrix_output = f"""
{stage}
- Throughput tokens per second: {self.throughput_tokens_per_second:.3f}
- TTFT: {self.ttft_ms:.3f} ms
- TPOT: {self.tpot_ms:.3f} ms
- RPS: {self.rps:.3f}
- Waiting queue number: {self.waiting_queue_num}
- Active requests number: {self.active_requests_num}"""
        print("-" * 20 + metrix_output + "\n" + "-" * 20)
