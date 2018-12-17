import chainer
from chainer.training import trigger as trigger_module

class FailureDetector(chainer.training.Extension):
    '''An object to Detect Local computing resource failure

    Check the tick of each iteration'''
    def __init__(self, comm):
        self.comm = comm

    def __call__(self, _trainer):
        self.tick()
        # Check the communicator still alive and not in suspention mode?
        self.comm.tick()

    def finalize(self):
        pass

    def initialize(self):
        pass

    def serialize(self):
        pass

    def on_error(self, trainer, exc, tb):
        pass

    def tick(self):
        pass


class Lineage(chainer.training.Extension):
    def __init__(self, comm, trigger=(1, 'epoch')):
        self.log_report = chainer.training.extensions.LogReport(trigger=trigger)
        self.comm = comm
        self.trigger = trigger # extention class property
        self.comm.load_state_reliable("lineage", self.log_report)
        print("first lineage!!!!")

    def __call__(self, trainer):
        self.log_report(trainer)
        self.comm.save_state_reliable("lineage", self.log_report)

    def serialize(self, serializer):
        self.log_report.serialize(serializer)

    @property
    def log(self):
        return self.log_report.log

        '''
    @property
    def name(self):
        return "Lineage"
        '''
