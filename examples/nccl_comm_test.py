import argparse
import chainer
import cupy
import numpy as np
from cupy.cuda import nccl
from echainer import NcclCommunicator, MinMaxPolicy
import json

def main():
    parser = argparse.ArgumentParser(description='ChainerMN example: MNIST')
    parser.add_argument('--late', '-l', action='store_true',
                        help='late join to the ring')
    parser.add_argument('--np', '-n', type=int, required=True,
                        help='Number of processes')
    parser.add_argument('--bind', '-p', type=str, required=True,
                        help='address to bind gRPC server')
    parser.add_argument('--etcd', '-c', type=str,
                        help='etcd location')
    # parser.add_argument('--rank', '-r', type=int)
    args = parser.parse_args()

    n = args.np
    bind = args.bind
    policy = MinMaxPolicy(n, n, block=True)

    def uid_gen(intra_rank):
        chainer.cuda.get_device_from_id(intra_rank).use()
        return json.dumps(nccl.get_unique_id())
    comm = NcclCommunicator(policy=policy, bind=bind)
    print(comm.intra_rank)

    nccl_comm = policy.nccl_comm
    stream = cupy.cuda.Stream.null

    a = cupy.ndarray([2,2,2], dtype=np.float32)
    b = cupy.ndarray([2,2,2], dtype=np.float32)

    a.real[:] = 2
    print(a)

    nccl_comm.allReduce(a.data, b.data, 8, nccl.NCCL_FLOAT32,
                        nccl.NCCL_SUM, stream.ptr)
    print(b)

    a.real[:] = comm.rank
    print('before bcast>', a)
    nccl_comm.bcast(a.data, 8, nccl.NCCL_FLOAT32, 0, stream.ptr)
    print('after bcast>', a)

    a.real[:] = comm.rank
    print('allreduce_obj', a)
    c = comm.allreduce_obj(a)
    print('after allreduce_obj', c)
    comm.leave()


if __name__ == '__main__':
    main()
