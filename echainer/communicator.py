from abc import ABCMeta
from abc import abstractmethod
import io
import json
import os
import pickle
import sys
import time
import threading
import traceback

import chainer
from chainer.training import extension

import chainermn
import six

import echainer_internal
from echainer import ticker
from echainer import _memory_utility

class ClusterUpdatedException(Exception):
    def __init__(self):
        pass

class ReformHandler(chainer.training.Extension):
    def __init__(self):
        pass

    def on_error(self, trainer, exc, tb):
        print("Error Handler!", exc)

    def finalize(self):
        print("finalizing ReformHandler...")

class CommunicatorResource(six.with_metaclass(ABCMeta)):
    def __init__(self):
        pass

    @abstractmethod
    def make_sesame(self, intra_rank):
        raise NotImplementedError()

    @abstractmethod
    def construct(self, size, sesame, rank):
        '''Do some user-land system construction. This will be called when the
        state of cluster is reset, actually after ok2fun() returned
        ('ok', <list>) and other resorces initialized.

        During this call no communication via communicator is available.
        '''
        raise NotImplementedError()

    @abstractmethod
    def destroy(self):
        '''Do user-land destruction. Will be called concurrently, different
        thread from main. e.g. call nccl_comm.destroy().

        '''
        raise NotImplementedError()

class MetaCommunicator(chainermn.CommunicatorBase):
    def __init__(self, policy, etcd='etcd://localhost:2379/hogehoge',
                 bind='localhost:20001', resources=[]):
        super(MetaCommunicator, self).__init__()
        self.etcd = etcd
        self.bind = bind
        self.policy = policy
        # must be instance of subclass of CommunicatorResource
        self.resources = resources

        '''uid_gen will be called in all processes and is guaranteed that
        the only one return value in consensus will be passed to Policy's
        ``construct()`` method. Example::

        def uid_gen(intra_rank):
            chainer.cuda.get_device_from_id(intra_rank).use()
            return nccl.get_unique_id()
        '''
        self.uid_mine = False

        ## Block here for cluster completion
        self.membership = echainer_internal.Membership(self.bind, self.etcd)
        self.membership.set_policy(self.policy)
        assert self.membership.rendezvous()

        self.monitor = ticker.SelfMonitor()

        # TODO: let it destroy once membership requests destruction
        self.destroyer_thread = threading.Thread(target=self.destroyer)
        self.destroyer_thread.start()

        self.registered_state = {}
        while not self.do_construct({}):
            time.sleep(1)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.leave()

    def get_monitor(self):
        return self.monitor

    def destroyer(self):
        self.run_destroyer = True
        while self.run_destroyer:
            destroy_done = self.membership.maybe_destroy()
            if destroy_done:
                for resource in self.resources:
                    try:
                        resource.destroy()
                    except Exception as e:
                        print("Failed to destroy resource @%s :" % self.bind, e)
            #if not self.monitor.is_alive():
            #    # TODO: exit gracefully
            #    print('monitor: false. Exiting')
            #    os._exit(1)
            time.sleep(1) # TODO: use mutex and condition variable

    def leave(self):
        # print('leaving')
        self.membership.leave()
        self.run_destroyer = False
        self.destroyer_thread.join()
        print('left the cluster')

    def register_state(self, name, obj):
        self.registered_state[name] = obj

    def fetch_state(self, name, obj):
        ret = self.membership.fetch_state(name)
        #print(ret)
        if ret is not None:
            (iteration, epoch, buf) = ret
            chainer.serializers.load_npz(io.BytesIO(buf), obj)
            return (iteration, epoch)

    def sync_cluster(self, optimizers):
        while True:
            while self.membership.maybe_update_cluster():
                # print("syncing cluster...")
                time.sleep(1)

            if self.do_construct(optimizers):
                return

    def save_all_states(self, iteration, epoch):
        for name in self.registered_state.keys():
            # print('saving state', name)
            buf = io.BytesIO()
            obj = self.registered_state[name]
            chainer.serializers.save_npz(buf, obj)
            self.membership.register_state(iteration, epoch, name, buf.getbuffer())

    def save_state_reliable(self, name, target):
        buf = io.BytesIO()
        chainer.serializers.save_npz(buf, target)
        self.membership.put_sesame(name, buf.getbuffer())
        # print(len(buf.getbuffer()), 'bytes saved to etcd')

    def load_state_reliable(self, name, target):
        buf = self.membership.get_sesame(name)
        if buf is not None:
            chainer.serializers.load_npz(io.BytesIO(buf), target)
            # print(len(buf), 'bytes loaded from etcd')

    def do_construct(self, optimizers):
        print("do_construct>", self.rank, self.intra_rank, len(self.resources))

        try:
            sesames0 = [resource.make_sesame(self.intra_rank)
                       for resource in self.resources]
            sesames = self.bcast_obj(sesames0)
            # TODO: debug log
            #if uid is not None and uid0 is not None:
            #print("sesame:", sesames0[:8], "=>", uid[:8], self.rank)
            for sesame, resource in zip(sesames, self.resources):
                resource.construct(self.size, sesame, self.rank)
            # self.policy.construct(self.size, uid, self.rank)
            return True
        except Exception as e:
            # Construct & make_sesame involves user code, thus
            # arbitrary exception may be raised.
            print("Failed to construct... ", e)
            traceback.print_exc()
            return False

    def get_progress_updater(self):
        return None

    @property
    def initial(self):
        return self.membership.is_initial()

    def set_initial(self, is_initial):
        self.membership.set_initial(is_initial)

    def get_uninitializer(self):
        @extension.make_extension(trigger=(1, 'iteration'))
        def uninitializer(_trainer):
            self.set_initial(False)
            # print('calling uninitializer', self.initial)
            # sys.exit(1)
        return uninitializer

    @property
    def rank(self):
        return self.membership.get_rank()

    @property
    def size(self):
        return self.membership.get_size()

    @property
    def intra_rank(self):
        return self.membership.get_intra_rank()

    @property
    def intra_size(self):
        return self.membership.get_intra_size()

    @property
    def inter_rank(self):
        '''The rank of this node in the cluster.'''
        return self.rank / self.intra_size

    @property
    def inter_size(self):
        return self.size / self.intra_size

    def split(self, color, key):
        raise NotImplementedError()

    def alltoall(self, xs):
        raise NotImplementedError()

    # on ndarrays and such
    def send(self, data, dest, tag):
        raise NotImplementedError()

    def recv(self, source, tag):
        raise NotImplementedError()

    def bcast(self, data, max_buf_len=None, root=0):
        raise NotImplementedError()

    def gather(self, data, root=0):
        raise NotImplementedError()

    def allgather(self, x):
        raise NotImplementedError()

    def allreduce(self, data):
        raise NotImplementedError()

    # on objects
    def send_obj(self, obj, dest, _tag=None):
        # print('send_obj to', dest)
        if self.membership.maybe_update_cluster():
            raise ClusterUpdatedException()
        buf = pickle.dumps(obj)
        self.membership.send(dest, buf)

    def recv_obj(self, source, _tag=None):
        # print('recv_obj from', source)
        if self.membership.maybe_update_cluster():
            raise ClusterUpdatedException()
        buf = self.membership.recv(source)
        if buf is None:
            raise echainer_internal.CommException()
        try:
            return pickle.loads(buf)
        except Exception as e:
            # _pickle.UnpicklingError: invalid load key, '\x00'.
            raise echainer_internal.CommException()

    def bcast_obj(self, obj, max_buf_len=None, root=0):
        if self.membership.maybe_update_cluster():
            raise ClusterUpdatedException()

        # VERY SLOW I KNOW; should be overrided
        if self.rank == root:
            for i in range(self.size):
                if i != root:
                    self.send_obj(obj, i)
            return obj
        else:
            return self.recv_obj(root)

    def gather_obj(self, obj, root=0):
        if self.membership.maybe_update_cluster():
            raise ClusterUpdatedException()
        if self.rank == root:
            for i in range(self.size):
                if self.rank != root:
                    yield self.recv_obj(i)
                else:
                    yield obj
        else:
            self.send_obj(obj, root)

    def allreduce_obj(self, obj):
        if self.membership.maybe_update_cluster():
            raise ClusterUpdatedException()
        # self.membership.maybe_update_cluster(False)

        import numpy
        ## TODO: how can copying to host memory avoided?
        xp = chainer.backend.get_array_module(obj)
        if xp != numpy:
            nobj = obj.get()
        else:
            nobj = obj

        right = (self.rank + 1) % self.size
        left = (self.rank - 1) % self.size
        basket = nobj

        for i in range(self.size - 1):
            self.send_obj(basket, right)
            lhs = self.recv_obj(left)
            basket = lhs + nobj

        if xp != numpy:
            return xp.asarray(basket)
        else:
            return basket

    # Special communication methods on grads and data of models
    def bcast_data(self, model):
        '''Broadcast Chainer model parameter data'''
        print('bcast_data via gRPC')
        if self.membership.maybe_update_cluster():
            raise ClusterUpdatedException()
        for _, param in sorted(model.namedparams()):
            if param.data is not None:
                buf = _memory_utility.get_device_memory_pointer(param.data)
                self.membership.bcast(buf, 0)

    def broadcast_data(self, model):
        self.bcast_data(model)

    def allreduce_grad(self, model):
        # print('allreduce_grad via gRPC')
        if self.membership.maybe_update_cluster():
            raise ClusterUpdatedException()
        for param in _memory_utility.extract_params_set_grad(model):
            # buf = _memory_utility.array_to_buffer_object(param.grad)
            buf = _memory_utility.get_device_memory_pointer(param.grad)
            self.membership.allreduce(buf)
            param.grad /= self.size
