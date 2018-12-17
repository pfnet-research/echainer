import sys
import time

import numpy as np

from echainer import MetaCommunicator, MinMaxPolicy

if __name__ == '__main__':
    bind = sys.argv[1]
    if len(sys.argv) > 2:
        repeat = int(sys.argv[2])
    else:
        repeat = 1000
    n = 2
    comm = MetaCommunicator(policy=MinMaxPolicy(n, n, block=True), bind=bind, unique_id="learn or die")
    print(comm, "me is", comm.rank, "size =", comm.size)
    count = 0
    while count < repeat:
        if comm.rank == 0:
            next = (comm.rank + 1) % comm.size
            print("Sending", count, "to", next)
            comm.send_obj( (count, 'mesg from %d to %d' % (comm.rank, next)), next)

        prev = (comm.rank - 1 + comm.size) % comm.size
        print("Recving", count, "from", prev)
        try:
            obj = comm.recv_obj(prev)
            print(obj)
        except Exception as e:
            print("execption~~~~~~~~~", e)
            continue

        if comm.rank > 0:
            next = (comm.rank + 1) % comm.size
            print("Sending", count, "to", next)
            comm.send_obj( (count, 'mesg from %d to %d' % (comm.rank, next)), next)

        count += 1

        s = comm.allreduce_obj(3)
        print(s, 3 * comm.size)

        a = np.random.rand(4, 4).astype(np.float32)
        print(a)
        comm.membership.allreduce(a)
        print(a)
        # time.sleep(0.1)

    print("broadcast")
    data = [1, 2, 3, 3, comm.rank]
    print(data)
    data = comm.bcast_obj(data)
    print(data)

    comm.leave()
