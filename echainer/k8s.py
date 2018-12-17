import subprocess as sb
from operator import itemgetter
from itertools import chain


def check_kubectl():
    try:
        res = sb.run('kubectl', capture_output=True)
        return res.returncode == 0
    except FileNotFoundError:
        return False

def parse_pod_owide(s):
    # print('original:', s)
    return [line.split() for line in s.strip().split('\n')[1:]]

def get_pod_owide():
    '''Make a list that maps ip address -> hostname'''
    res = sb.run(['kubectl', 'get', 'pod', '-owide'], capture_output=True)
    assert res.returncode == 0
    return parse_pod_owide(res.stdout.decode("utf-8").strip())

def sort_node_list(hosts, table=None):
    if table is None:
        table = get_pod_owide()
        # print('parsed:', table)
    d = {}
    for line in table:
        # ip => host, hosts
        d[line[5]] = (line[6], [])
    pairs = []
    for host in hosts:
        ip = host.split(':')[0]
        d[ip][1].append(host)
    pairs = []
    for key in sorted(d.keys()):
        # list of host, hosts
        if len(d[key]) > 0:
            pairs.append(d[key])

    sorted_hosts = [proc for proc in [sorted(procs) for (host, procs) in sorted(pairs)]]
    sorted_hosts = list(chain.from_iterable(sorted_hosts))
    # print("list sorted:", sorted_hosts)

    return sorted_hosts
