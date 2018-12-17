import abc
from abc import abstractmethod
import time

import chainer
import six

from echainer import k8s


class ScalePolicy(six.with_metaclass(abc.ABCMeta)):
    '''It can contain user-side resources. Or we don't need it?
    '''
    @abstractmethod
    def ok2run(self, hosts, initial):
        '''Is it OK to run the computation?

        Input is the potential list of next ...  This function MAY NOT
        change any local state as it may be called any arbitrary
        timing by the meta communicator. Thus this method should not
        have any side effects.

        The list ``hosts`` is NOT sorted by sort_node_list.

        '''
        # assert 0 <= rank and rank < size

        raise NotImplementedError()

    ## @abstractmethod
    def sort_node_list(self, hosts):
        '''Sort the node list for optimization, must be consistent. or not?

        Recuired for NCCL w/InfiniBand configuration, not sure on Ethernet-based networks.

        TODO: Maybe it is too much, to add sorter method here, forcing
        all descendant classes. This is maybe because currently we know
        only two use cases:
        1. The host list may be sorted according to the original hostname or
           IP addresses, and the IB physical topology is same as it.
        2. Kubernetes assigns randomly ordered IP-address to its Pods, thus
           the order of ``hosts`` might not reflect physical topology of IB
           network.

        '''
        return sorted(hosts)


class FailStop(ScalePolicy):
    def __init__(self, n):
        assert n > 0
        self.n = n

    def ok2run(self, hosts, initial):
        '''Fail stop policy'''
        if len(hosts) == self.n:
            return 'ok'
        elif initial:
            return 'wait'
        else:
            return 'fail'


class MinMaxPolicy(ScalePolicy):
    def __init__(self, n_min, n_max, block=False):
        assert 0 < n_max
        assert 0 < n_min
        assert n_min <= n_max
        self.n_max = n_max
        self.n_min = n_min
        self.block = block
        self.hosts = []


    def ok2run(self, hosts, initial): # => 'wait' | 'fail' | ('ok', needs_reconstruction)
        size = len(hosts)
        print("ok2run:", hosts)
        if initial:
            if len(hosts) == self.n_max:
                # return ('ok', hosts)
                return 'ok'
            else:
                return 'wait'

        if size < self.n_min or self.n_max < size:
            if self.block:
                return 'wait'
            return 'fail'

        # Host list unchanged, hopefully. It's noop or so
        # If it's during rendez-vous, original host list will be used.
        return 'ok'

    def sort_node_list(self, hosts):
        new_hosts = k8s.sort_node_list(hosts)
        return new_hosts
        # return sorted(hosts)

class MinMaxPolicy2(ScalePolicy):
    def __init__(self, n_min, n_max, n_start):
        assert 0 < n_max and 0 < n_min
        assert 0 < n_start
        assert n_min <= n_start and n_start <= n_max
        self.n_max = n_max
        self.n_min = n_min
        self.n_start = n_start

    def sort_node_list(self, hosts):
        return sorted(hosts)
