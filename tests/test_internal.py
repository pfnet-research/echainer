from itertools import product
import random

import echainer_internal
from echainer import k8s

def dont_test_internal_yet():
    membership = echainer_internal.Membership("localhost:12345", "file:///tmp/echainer-membership")
    membership.rendezvous()


def test_k8s_sorter():
    dummy = '''
NAME               READY     STATUS    RESTARTS   AGE       IP               NODE
echainer-dev-csst2   1/1       Running   0          5m        172.23.243.84    gpcl2-gpu108
echainer-dev-l528z   1/1       Running   0          5m        172.23.220.6     gpcl2-gpu110
echainer-etcd        1/1       Running   0          5m        172.23.245.183   gpcl2-cpu020
'''
    table = k8s.parse_pod_owide(dummy)
    ips = ['172.23.243.84', '172.23.220.6']
    ports = range(11111, 11120)
    hosts = [ '%s:%d' % (ip, port) for ip, port in product(ips, ports)]
    shuffled = random.sample(hosts, len(hosts))
    ans = hosts
    hosts = k8s.sort_node_list(shuffled, table)
    print(ans)
    print(hosts)
    assert len(ans) == len(list(hosts))
    for a, h in zip(ans, hosts):
        assert a == h
