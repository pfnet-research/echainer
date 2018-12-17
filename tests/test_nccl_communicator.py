import os
import pytest
import numpy

import chainer
from chainer.testing.attr import gpu

from echainer import policies
from echainer import NcclCommunicator

@gpu
@pytest.fixture(scope='module')
def comm(request):
    size = request.config.getoption("--size")
    intra_rank = request.config.getoption("--intra_rank")
    policy = policies.FailStop(size)
    port = 11111 + intra_rank
    if os.getenv('ETCDIP'):
        etcd = 'etcd://%s:2379/echainer-testing' % os.getenv('ETCDIP')
    else:
        etcd = 'etcd://localhost:2379/echainer-testing'
    print('connecting...')
    bind = 'localhost:%d' % port
    with NcclCommunicator(policy, etcd, bind) as comm:
        assert size == comm.size
        assert intra_rank == comm.intra_rank
        yield comm

@gpu
def test_send_recv(comm):

    next = (comm.rank + 1) % comm.size
    prev = (comm.rank + comm.size - 1) % comm.size
    comm.send_obj( ('from', comm.rank, 'to', next), next )
    res = comm.recv_obj(prev)
    assert 'from' == res[0]
    assert prev == res[1]
    assert 'to' == res[2]
    assert comm.rank == res[3]

@gpu
def test_bcast_obj(comm):
    for root in range(comm.size):
        if root == comm.rank:
            msg = ('the root is', root)
        else:
            msg = None
        res = comm.bcast_obj(msg, root=root)
        assert len(res) == 2
        assert res[1] == root

@gpu
def test_gather_obj(comm):
    for root in range(comm.size):
        msg = ('msg from', root)
        res = comm.gather_obj(msg, root=root)
        if comm.rank == root:
            res = list(res)
            for m, r in res:
                assert m == 'msg from'
                assert r == root
            assert len(list(res)) == comm.size
@gpu
def test_allreduce_obj(comm):
    ans = sum(i for i in range(comm.size))
    s = comm.allreduce_obj(comm.rank)
    assert ans == s

    a = numpy.random.rand(16, 16, 16)
    s = comm.allreduce_obj(a)
    assert a.shape == s.shape
    assert numpy.all(a < s)

class ExampleModel(chainer.Chain):
    def __init__(self, dtype=None):
        if dtype is not None:
            self.dtype = dtype
        super(ExampleModel, self).__init__()
        with self.init_scope():
            self.a = chainer.links.Linear(2, 3)
            self.b = chainer.links.Linear(3, 4)
            self.c = chainer.links.Linear(None, 5)

@gpu
def test_bcast_data(comm):
    model = ExampleModel()
    model.to_gpu(device=comm.intra_rank)

    model.a.W.data[:] = comm.rank
    model.b.W.data[:] = comm.rank + 1
    model.c.b.data[:] = comm.rank + 2
    comm.bcast_data(model)

    chainer.testing.assert_allclose(model.a.W.data, 0 * numpy.ones((3, 2)))
    chainer.testing.assert_allclose(model.b.W.data, 1 * numpy.ones((4, 3)))
    chainer.testing.assert_allclose(model.c.b.data, 2 * numpy.ones((5, )))

@gpu
def test_allreduce_grad(comm):
    model = ExampleModel()
    model.to_gpu(device=comm.intra_rank)
    # We need to repeat twice for regressions on lazy initialization of
    # sub comms.
    for _ in range(2):
        model.a.W.grad[:] = comm.rank
        model.b.W.grad[:] = comm.rank + 1
        model.c.b.grad[:] = comm.rank + 2

        comm.allreduce_grad(model)
        base = (comm.size - 1.0) / 2

        chainer.testing.assert_allclose(model.a.W.grad,
                                        (base + 0) * numpy.ones((3, 2)))
        chainer.testing.assert_allclose(model.b.W.grad,
                                        (base + 1) * numpy.ones((4, 3)))
        chainer.testing.assert_allclose(model.c.b.grad,
                                        (base + 2) * numpy.ones((5, )))
