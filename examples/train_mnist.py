#!/usr/bin/env python
from __future__ import print_function

import argparse
import json
import time

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

import chainermn
import echainer
from echainer import MetaCommunicator, ReformHandler, ClusterUpdatedException, CommException
from echainer.policies import MinMaxPolicy, MinMaxPolicy2

class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(784, n_units),  # n_in -> n_units
            l2=L.Linear(n_units, n_units),  # n_units -> n_units
            l3=L.Linear(n_units, n_out),  # n_units -> n_out
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


def main():
    parser = argparse.ArgumentParser(description='ChainerMN example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', action='store_true',
                        help='Use GPU')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--np', '-n', type=int, required=True,
                        help='Minimum number of processes')
    parser.add_argument('--bind', '-p', type=str, required=True,
                        help='address to bind gRPC server')
    parser.add_argument('--etcd', '-c', type=str,
                        default='etcd://127.0.0.1:2379/train_mnist.py',
                        help='etcd location and path')
    args = parser.parse_args()

    # Prepare ChainerMN communicator.
    n = args.np
    bind = args.bind
    scale_policy = MinMaxPolicy(n, n, block=True)
    comm = None
    if args.gpu:
        from echainer import NcclCommunicator
        comm = NcclCommunicator(policy=scale_policy, bind=bind,
                                etcd=args.etcd)
    else:
        comm = MetaCommunicator(policy=scale_policy, bind=bind,
                                etcd=args.etcd)

    late = not comm.initial

    if comm.rank == 0:
        print('==========================================')
        print('Num process (COMM_WORLD): {}'.format(comm.size))
        if args.gpu:
            print('Using GPU ', comm.intra_rank)
        print('Num unit: {}'.format(args.unit))
        print('Num Minibatch-size: {}'.format(args.batchsize))
        print('Num epoch: {}'.format(args.epoch))
        print('==========================================')

    device = -1
    model = L.Classifier(MLP(args.unit, 10))
    if args.gpu:
        device = comm.intra_rank
        model.to_gpu(device=device)
    print('Using GPU ', device)

    # Create a multi node optimizer from a standard Chainer optimizer.
    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.Adam(), comm)
    optimizer.setup(model)
    done = False
    retry = False
    while not done:
        if args.gpu and retry:
            device = comm.intra_rank
            print('Using GPU No.', comm.intra_rank)
            model.to_gpu(device=device)

            optimizer = chainermn.create_multi_node_optimizer(
                chainer.optimizers.Adam(), comm)
            optimizer.setup(model)

        # Split and distribute the dataset. Only worker 0 loads the whole dataset.
        # Datasets of worker 0 are evenly split and distributed to all workers.
        print('get dataset')
        if comm.rank == 0:
            train, test = chainer.datasets.get_mnist()
        else:
            train, test = None, None

        print('scatter dataset')
        train = chainermn.scatter_dataset(train, comm, shuffle=True)
        test = chainermn.scatter_dataset(test, comm, shuffle=True)

        print('create iterator')
        train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
        test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                     repeat=False, shuffle=False)

        updater = training.StandardUpdater(train_iter, optimizer, device=device)
        trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

        # Create a multi node evaluator from a standard Chainer evaluator.
        evaluator = extensions.Evaluator(test_iter, model, device=device)
        evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
        trainer.extend(evaluator)

        trainer.extend(ReformHandler())
        trainer.extend(comm.get_monitor())
        trainer.extend(comm.get_uninitializer(), trigger=(1, 'iteration'))

        # Some display and output extensions are necessary only for one worker.
        # (Otherwise, there would just be repeated outputs.)
        if comm.rank == 0:
            trainer.extend(extensions.dump_graph('main/loss'))
            trainer.extend(echainer.extension.Lineage(comm))
            trainer.extend(extensions.PrintReport(
                ['epoch', 'main/loss', 'validation/main/loss',
                 'main/accuracy', 'validation/main/accuracy', 'elapsed_time'],
                log_report='Lineage'))
            trainer.extend(extensions.ProgressBar())

        # Register extension to save trainer's progress (iteration) in communicator
        # trainer.extend(comm.get_progress_updater())

        if args.resume:
            chainer.serializers.load_npz(args.resume, trainer)

        # Optimizer includes model parameters and other params in optimizer
        comm.register_state('optimizer', optimizer)
        comm.register_state('model', model)
        # Iterators: Well if number of nodes changed then current
        # position becomes wrong but try to recover That's why
        # recoevering iterators are nonsense.
        # Trainer: Too large and it includes Iterators.
        print(updater.epoch, updater.iteration)

        if retry or late:
            (iteration, epoch) = comm.fetch_state('optimizer', optimizer)
            (iteration, epoch) = comm.fetch_state('model', model)
            train_iter.epoch = epoch
            updater.iteration = iteration

        optimizers = trainer.updater.get_all_optimizers()
        # bcast again anyway
        for name in optimizers.keys():
            optimizers[name].reset_prev_params()

        try:
            print('start trainer.run(), ', trainer.updater.iteration, trainer.updater.epoch)
            trainer.run()
            done = trainer._done
        except CommException as ce:
            print(">>>>>>>>>>>", ce, updater.iteration, updater.epoch)
            comm.save_all_states(updater.iteration, updater.epoch)
            # Here comm will be ready to accept fetch state calls and once all
            # nodes got catched up it'll return and continue to run: TODO
            comm.sync_cluster(trainer.updater.get_all_optimizers())
            retry = True
            continue
        except ClusterUpdatedException as ce:
            print(">>>>>>>>>>>", ce)
            comm.save_all_states(updater.iteration, updater.epoch)
            comm.sync_cluster(trainer.updater.get_all_optimizers())
            retry = True
            continue
        except Exception as e:
            print("Unexpected >>>>>>>>>>>", e)
            break

    # TODO: this should be called cleanly, unless it runs forever somehow...
    comm.leave()


if __name__ == '__main__':
    main()
