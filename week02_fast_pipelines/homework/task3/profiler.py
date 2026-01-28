import json
import time
import torch
import os
from collections import defaultdict


class Profile:
    def __init__(self, model, name="model", wait=0, warmup=None, active=None, repeat=1):
        assert warmup is not None and active is not None
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.current_step = 0
        self.current_repeating = 0

        self.name_map = self._build_name_map(model, name)

        self.global_start = torch.cuda.Event(enable_timing=True)
        self.global_start.record()

        self.model = model
        self.events = []
        self.events_list = []

        self.cuda_event_starts_forward = {}
        self.cuda_event_ends_forward = {}
        self.cuda_event_starts_backward = {}
        self.cuda_event_ends_backward = {}
        self.is_registered_cuda_event_starts_forward = {}
        self.is_registered_cuda_event_ends_forward = {}
        self.is_registered_cuda_event_starts_backward = {}
        self.is_registered_cuda_event_ends_backward = {}

        self.forward_times = {}
        self.backward_times = {}

        for full_name, module in model.named_modules():
            module.register_forward_pre_hook(self._forward_pre_hook)
            module.register_forward_hook(self._forward_post_hook)
            module.register_full_backward_pre_hook(self._backward_pre_hook)
            module.register_full_backward_hook(self._backward_post_hook)

        self.update_events()
    
    def update_events(self):
        if self.current_step >= self.wait and self.current_repeating < self.repeat:
            for full_name, module in self.model.named_modules():
                self.cuda_event_starts_forward[module] = torch.cuda.Event(enable_timing=True)
                self.cuda_event_ends_forward[module] = torch.cuda.Event(enable_timing=True)
                self.cuda_event_starts_backward[module] = torch.cuda.Event(enable_timing=True)
                self.cuda_event_ends_backward[module] = torch.cuda.Event(enable_timing=True)
                self.is_registered_cuda_event_starts_forward[module] = False
                self.is_registered_cuda_event_ends_forward[module] = False
                self.is_registered_cuda_event_starts_backward[module] = False
                self.is_registered_cuda_event_ends_backward[module] = False
    
    def _build_name_map(self, model, name="model"):
        name_map = {}
        for full_name, module in model.named_modules():
            if full_name == "":
                full_name = name

            if self._is_leaf(module):
                name_map[module] = module.__class__.__name__
            else:
                name_map[module] = f"{full_name}: {module.__class__.__name__}"

        return name_map

    def _is_leaf(self, module):
        return len(list(module.children())) == 0

    def _forward_pre_hook(self, module, inputs):
        if self.current_step >= self.wait and self.current_repeating < self.repeat:
            if not self.is_registered_cuda_event_starts_forward[module]:
                self.cuda_event_starts_forward[module].record()
                self.is_registered_cuda_event_starts_forward[module] = True

    def _forward_post_hook(self, module, inputs, outputs):
        if self.current_step >= self.wait and self.current_repeating < self.repeat:
            if not self.is_registered_cuda_event_ends_forward[module]:
                self.cuda_event_ends_forward[module].record()
                self.is_registered_cuda_event_ends_forward[module] = True

    def _backward_pre_hook(self, module, grad_output):
        if self.current_step >= self.wait and self.current_repeating < self.repeat:
            if not self.is_registered_cuda_event_starts_backward[module]:
                self.cuda_event_starts_backward[module].record()
                self.is_registered_cuda_event_starts_backward[module] = True

    def _backward_post_hook(self, module, grad_input, grad_output):
        if self.current_step >= self.wait and self.current_repeating < self.repeat:
            if not self.is_registered_cuda_event_ends_backward[module]:
                self.cuda_event_ends_backward[module].record()
                self.is_registered_cuda_event_ends_backward[module] = True

    def __enter__(self):
        self.current_step = 0
        self.current_repeating = 0
        return self
 
    def __exit__(self, type, value, traceback):
        self.summary()

    def _process_events(self):
        pending_events = []
        for event in self.events_list:
            if event["event"].query():
                self.events.append({
                    "name": event["name"] + ("forward" if "forward" in event["type"] else "backward"),
                    "ph": "B" if "start" in event["type"] else "E",
                    "ts": self.global_start.elapsed_time(event["event"]) * 1000,
                    "pid": 0,
                    "tid": 0,
                })
            else:
                pending_events.append(event)
        self.events_list = pending_events
        return len(self.events_list) == 0
    
    def step(self):
        if self.current_step >= self.wait + self.warmup and self.current_repeating < self.repeat:
            for _, module in self.model.named_modules():
                if self.is_registered_cuda_event_starts_forward[module]:
                    self.events_list.append({"name": self.name_map[module], "type": "forward_start", "event": self.cuda_event_starts_forward[module]})
                if self.is_registered_cuda_event_ends_forward[module]:
                    self.events_list.append({"name": self.name_map[module], "type": "forward_end", "event": self.cuda_event_ends_forward[module]})
                if self.is_registered_cuda_event_starts_backward[module]:
                    self.events_list.append({"name": self.name_map[module], "type": "backward_start", "event": self.cuda_event_starts_backward[module]})
                if self.is_registered_cuda_event_ends_backward[module]:
                    self.events_list.append({"name": self.name_map[module], "type": "backward_end", "event": self.cuda_event_ends_backward[module]})
            self.cuda_event_starts_forward = {}
            self.cuda_event_ends_forward = {}
            self.cuda_event_starts_backward = {}
            self.cuda_event_ends_backward = {}
        self._process_events()
        self.current_step += 1
        if self.current_step >= self.wait + self.warmup + self.active:
            self.current_repeating += 1
            self.current_step = 0
        self.update_events()
        

    def summary(self):
        if not self._process_events():
            print('Not all events have finished, you might want to do cuda.synchronize() to get full results')
        print("Summary:")
        print('Total events handeled: ', len(self.events))

    def to_perfetto(self, path="trace.json"):
        if not self._process_events():
            print('Not all events have finished, you might want to do cuda.synchronize() to get full results')
        with open(path, 'w') as f:
            json.dump(self.events, f)
