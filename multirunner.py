#!/usr/bin/env python3
'''Multirunner
===========

For people who's ``mpirun`` is too much and too short; this just spawns
multiple processes locally with fault tolerance and argument formatting.

Usage Examples
--------------

::

  $ multirunner.py --np <num>  -- python3 .....

See ``multirunner.py --help`` for more detail usage.

References
----------

Distributed XGBoost YARN on AWS
http://xgboost.readthedocs.io/en/latest/tutorials/aws_yarn.html

dmlc_tracker.tracker.py, using ssh.py at the same directory
https://github.com/dmlc/dmlc-core/blob/master/tracker/dmlc_tracker/tracker.py

From here
https://github.com/apache/incubator-mxnet/blob/master/tools/launch.py#L111-L113

'''

import argparse
import os
import re
import socket
import subprocess as sb
import sys
import time
from typing import List, Dict

from jinja2 import Template

class NeedsWaitException(Exception):
    pass

def get_owide():
    cp = sb.run(['kubectl', 'get', 'pod', '-owide'], stdout=sb.PIPE)
    return cp.stdout.decode('utf-8').strip().split('\n')

def get_etcd_prefix(lines):
    # ETCDIP=`kubectl get pod -owide | grep echainer-etcd | awk '{print $6;}'`
    hosts = []
    for line in lines:
        tokens = line.split()
        if len(tokens) > 5 and tokens[0][:13] == 'echainer-etcd':
            if tokens[5] == '<none>' or tokens[2] != 'Running':
                raise NeedsWaitException()
            hosts.append('%s:%d' % (tokens[5], 2379))
    assert len(hosts) > 0
    addrs = ','.join(hosts)
    return 'etcd://%s' % addrs

def get_inter_size(lines):
    # MY_INTER_SIZE=`kubectl get pod -owide | grep echainer| grep -v etcd | wc | awk '{print $1;}'`
    hosts = []
    for line in lines:
        tokens = line.split()
        if len(tokens) > 5 and tokens[0][:8] == 'echainer' \
            and tokens[0][:13] != 'echainer-etcd':
            hosts.append(tokens[0])
    return len(hosts)

def get_gpu_size():
    # MY_WORKERS=`ls /dev/nvidia* | grep -e "\/dev\/nvidia[0-7]$" | wc -l`
    cp = sb.run(['ls', '/dev'], stdout=sb.PIPE)
    lines = cp.stdout.decode('utf-8').strip().split('\n')
    pattern = re.compile('^nvidia\d$')
    count = 0
    for line in lines:
        # print(line, "->")
        if pattern.match(line) is not None:
            print(line)
            count += 1
    return count

def format_cmd(cmd : List[str], format : Dict[str, str],
               rank : int, intra_rank : int, host : str) -> List[str]:
    assert format is not None
    assert cmd[0] == '--'
    m = format
    m['host'] = host
    m['intra_rank'] = intra_rank
    cmd2 = []
    for s in cmd[1:]:
        t = Template(s)
        cmd2.append(t.render(m))
    return cmd2

def format_env(env : Dict[str, str], format : Dict[str, str],
               rank : int, intra_rank : int,  host : str) -> Dict[str, str]:
    m = format
    m['host'] = host
    m['intra_rank'] = intra_rank
    e = {}
    for k, v in env.items(): # needs six for 2
        t = Template(str(v))
        e[k] = t.render(m)
    e['INTRA_RANK'] = str(intra_rank)
    e['HOST'] = host
    return e

def format_local_cmd(cmd : List[str], format : Dict[str, str], rank : int) -> List[str]:
    return format_cmd(cmd, format, rank, rank, socket.gethostname())

def format_local_env(env : Dict[str, str], format : Dict[str, str], rank : int) -> Dict[str, str]:
    return format_env(env, format, rank, rank, socket.gethostname())

def parse_props(pairs : List[str]) -> Dict[str, str]:
    props = {}
    if pairs is None:
        return props

    for kv in pairs:
        k, v = kv.split('=')
        props[k] = v
    return props

def local_run(np : int, cmd : str, format : Dict[str, str], env : Dict[str, str], dry_run : bool):
    procs = {}
    for i in range(np):
        # if hosts is empty, then run then all locally without paramiko/ssh
        # else, run then all out of paramiko/ssh! (TODO)
        command = format_local_cmd(cmd, format, i)
        environ = format_local_env(env, format, i)

        if dry_run:
            print("p", i, ":", ' '.join(command), environ)
        else:
            # TODO: output mode choice:
            # 0. all to current tty
            # 1. only for rank 0 to tty, others to /dev/null
            # 2. all to files
            # 3. rank 0 to tty, others to files
            # TODO: Know what happens when a child process is killed or failed early
            p = sb.Popen(command, env=environ)
            procs[i] = p

    if dry_run:
        return [0] * np

    for i in range(np):
        p = procs[i]
        code = p.wait()
        yield code

def main():
    parser = argparse.ArgumentParser(epilog='''\
MultiRunner

http://looneytunes.wikia.com/wiki/File:Road-runner-4.jpg

Run your command in parallel, controlling each context and environment variables.

''',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--np', '-n', type=int, help="Number of hosts")
    parser.add_argument('--dry-run', action='store_true',
                        help="Check all commands with its env")
    parser.add_argument('--env', '-e', type=str,
                        help="Environment variales, k1=v1 k2=v2. Can be formatted with format variables.",
                        nargs='*', default=[])
    parser.add_argument('--format', '-f', type=str,
                        help="Format variales, k1=v1 k2=v2",
                        nargs='*', default=[])
    parser.add_argument('--', dest='')
    parser.add_argument('cmd', nargs=argparse.REMAINDER,
                        help="""The command to run. You can use jinja2-style template, rendering
and tiny computation which is replaced by format variables set by
``--format``. It has implicit environments: {{intra_rank}}""")
    args = parser.parse_args()

    env = parse_props(args.env)
    parent_env = dict(os.environ)
    format = parse_props(args.format)

    etcd_prefix = None
    gpu_size = None
    retry = 0
    print('waiting for etcd comes up')
    while etcd_prefix is None or gpu_size is None:
        try:
            owide_lines = get_owide()
            etcd_prefix = get_etcd_prefix(owide_lines)
            print('etcd prefix:', etcd_prefix)
            # inter_size = get_inter_size(owide_lines)
            gpu_size = get_gpu_size()
            # print(inter_size, gpu_size)
            print(gpu_size, "GPUs found.")
        except NeedsWaitException as nwe:
            retry += 1
            time.sleep(3)
            print('retrying', retry)
            continue
        if retry > 20:
            print('etcd is not coming up after', 20, "retries")
            exit(-1)

    format['etcd_prefix'] = etcd_prefix
    format['num_workers'] = gpu_size
    if args.np is None:
        np = gpu_size
    else:
        np = args.np
                
    print(np, "processes to be launched.")
    return_codes = local_run(np, args.cmd, format, {**env, **parent_env}, args.dry_run)

    if all( [ code == 0 for code in return_codes ] ):
        return 0

    sys.stderr.write("Some procs failed %s\n" % return_codes)
    sys.stderr.flush()
    return -1

if __name__ == '__main__':
    main()
