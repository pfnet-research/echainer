import time
import random

from chainer.training import Extension
from echainer_internal import ThresholdTicker

class SelfMonitor(Extension):

    def __init__(self, timeout_ms):
        self.timeout_ms = timeout_ms
        self.started = False

    def __init__(self):
        self.started = False
    
    def initialize(self, _trainer):
        self.ticker = ThresholdTicker()
        if hasattr(self, 'timeout_ms'):
            self.ticker.set_int_param('timeout_ms', timeout_ms)

    def __call__(self, trainer):
        self.started = True
        self.ticker.tick()

    def finalize(self):
        pass

    def serialize(self, serializer):
        if hasattr(self, 'timeout_ms'):
            serializer('timeout_ms', self.timeout_ms)
            if not isinstance(serializer, serializer_module.Serializer):
                self.ticker.set_int_param('timeout_ms', self.timeout_ms)

    def is_alive(self):
        if self.started:
            return self.ticker.is_alive()
        else:
            '''Training hasn't started yet'''
            return True

class RandomLatency(Extension):
    def __init__(self, max_sleep_sec = 20):
        '''TODO: implement serialize'''
        self.max_sleep_sec = max_sleep_sec

    def __call__(self, _trainer):
        t = random.randint(0, self.max_sleep_sec)
        time.sleep(t)
