'''
For configuration:
https://github.pfidev.jp/sugoi-hayai/gc18/issues/60

For code:
https://github.com/chainer/chainercv/pull/436
'''

from __future__ import division
import argparse
import json
import multiprocessing

import chainer
from chainer.datasets import TransformDataset
from chainer import iterators
from chainer.links import Classifier
from chainer.optimizer import WeightDecay
from chainer.optimizers import CorrectedMomentumSGD
from chainer import training
from chainer.training import extensions

from chainercv.datasets import directory_parsing_label_names
from chainercv.datasets import DirectoryParsingLabelDataset
from chainercv.transforms import center_crop
from chainercv.transforms import random_flip
from chainercv.transforms import random_sized_crop
from chainercv.transforms import resize
from chainercv.transforms import scale

from chainercv.chainer_experimental.training.extensions import make_shift

from chainercv.links.model.resnet import Bottleneck
from chainercv.links import ResNet101
from chainercv.links import ResNet152
from chainercv.links import ResNet50

import chainermn
import echainer
from echainer import monkey_patch
from echainer import MetaCommunicator, ReformHandler, ClusterUpdatedException, CommException
from echainer.policies import MinMaxPolicy

import cv2
cv2.setNumThreads(2)

class TrainTransform(object):

    def __init__(self, mean):
        self.mean = mean

    def __call__(self, in_data):
        img, label = in_data
        _, H, W = img.shape
        img = random_sized_crop(img)
        img = resize(img, (224, 224))
        img = random_flip(img, x_random=True)
        img -= self.mean
        return img, label


class ValTransform(object):

    def __init__(self, mean):
        self.mean = mean

    def __call__(self, in_data):
        img, label = in_data
        img = scale(img, 256)
        img = center_crop(img, (224, 224))
        img -= self.mean
        return img, label


def main():
    archs = {
        'resnet50': {'class': ResNet50, 'score_layer_name': 'fc6',
                     'kwargs': {'arch': 'fb'}},
        'resnet101': {'class': ResNet101, 'score_layer_name': 'fc6',
                      'kwargs': {'arch': 'fb'}},
        'resnet152': {'class': ResNet152, 'score_layer_name': 'fc6',
                      'kwargs': {'arch': 'fb'}}
    }
    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('train', help='Path to root of the train dataset')
    parser.add_argument('val', help='Path to root of the validation dataset')
    parser.add_argument('--arch',
                        '-a', choices=archs.keys(), default='resnet50',
                        help='Convnet architecture')
    parser.add_argument('--loaderjob', type=int, default=4)
    parser.add_argument('--batchsize', type=int, default=32,
                        help='Batch size for each worker')
    parser.add_argument('--lr', type=float)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--out', type=str, default='result')
    parser.add_argument('--epoch', type=int, default=90)

    parser.add_argument('--min', type=int, required=True,
                        help='Minimum number of processes')
    parser.add_argument('--start', type=int, required=True,
                        help='Number of processes to start')
    parser.add_argument('--bind', '-p', type=str, required=True,
                        help='address to bind gRPC server')
    parser.add_argument('--etcd', '-c', type=str, default='etcd://localhost:2379/echainer-test',
                        help='etcd location')

    parser.add_argument
    args = parser.parse_args()

    # This fixes a crash caused by a bug with multiprocessing and MPI.
    multiprocessing.set_start_method('forkserver')
    p = multiprocessing.Process()
    p.start()
    p.join()

    scale_policy = MinMaxPolicy(args.min, args.start, block=True)
    # scale_policy = ScalePolicy2(args.min, args.min * 100, args.start, use_nccl=True)
    from echainer import NcclCommunicator
    comm = NcclCommunicator(policy=scale_policy, bind=args.bind, etcd=args.etcd)

    # comm = chainermn.create_communicator(args.communicator)
    device = comm.intra_rank

    label_names = directory_parsing_label_names(args.train)

    arch = archs[args.arch]
    extractor = arch['class'](n_class=len(label_names), **arch['kwargs'])
    extractor.pick = arch['score_layer_name']
    model = Classifier(extractor)
    # Following https://arxiv.org/pdf/1706.02677.pdf,
    # the gamma of the last BN of each resblock is initialized by zeros.
    for l in model.links():
        if isinstance(l, Bottleneck):
            l.conv3.bn.gamma.data[:] = 0

    done = False
    retry = False
    while not done:
        if args.lr is not None:
            lr = args.lr
        else:
            lr = 0.1 * (args.batchsize * comm.size) / 256
            if comm.intra_rank == 0:
                print('lr={}: lr is selected based on the linear '
                      'scaling rule'.format(lr))


        print("Loading dataset", comm.rank)
        if comm.rank == 0:
            train_data = DirectoryParsingLabelDataset(args.train)
            val_data = DirectoryParsingLabelDataset(args.val)
            train_data = TransformDataset(
                train_data, TrainTransform(extractor.mean))
            val_data = TransformDataset(val_data, ValTransform(extractor.mean))
            print('finished loading dataset len = ', len(train_data))
        else:
            train_data, val_data = None, None

        train_data = chainermn.scatter_dataset(train_data, comm, shuffle=True)
        val_data = chainermn.scatter_dataset(val_data, comm, shuffle=True)

        print('scatter done: len =', len(train_data), ' rank = ', comm.intra_rank)

        train_iter = iterators.SerialIterator(train_data,
                                              args.batchsize)
        val_iter = iterators.SerialIterator(val_data, args.batchsize,
                                            repeat=False, shuffle=False)

        '''
        train_iter = chainer.iterators.MultiprocessIterator(
            train_data, args.batchsize, shared_mem=3 * 224 * 224 * 4,
            n_processes=args.loaderjob)
        val_iter = iterators.MultiprocessIterator(
            val_data, args.batchsize,
            repeat=False, shuffle=False, shared_mem=3 * 224 * 224 * 4,
            n_processes=args.loaderjob)
        '''

        optimizer = chainermn.create_multi_node_optimizer(
            CorrectedMomentumSGD(lr=lr, momentum=args.momentum), comm)
        monkey_patch.patch_mn_optimizer(optimizer)
        optimizer.setup(model)
        for param in model.params():
            if param.name not in ('beta', 'gamma'):
                param.update_rule.add_hook(WeightDecay(args.weight_decay))

        #if retry:
        device = comm.intra_rank
        print('Using GPU No.', comm.intra_rank,
              'to_gpu: device =', device,
              'intra_rank =', comm.intra_rank,
              'rank =', comm.rank,
              'addr =', args.bind)
        model.to_gpu(device=device)
        #chainer.cuda.get_device_from_id(device).use()
        #model.to_gpu()

        # Configure GPU setting
        chainer.cuda.set_max_workspace_size(1 * 1024 * 1024 * 1024)
        chainer.using_config('autotune', True)

        updater = chainer.training.StandardUpdater(
            train_iter, optimizer, device=device)

        trainer = training.Trainer(
            updater, (args.epoch, 'epoch'), out=args.out)

        @make_shift('lr')
        def warmup_and_exponential_shift(trainer):
            epoch = trainer.updater.epoch_detail
            warmup_epoch = 5
            if epoch < warmup_epoch:
                if lr > 0.1:
                    warmup_rate = 0.1 / lr
                    rate = warmup_rate \
                           + (1 - warmup_rate) * epoch / warmup_epoch
                else:
                    rate = 1
            elif epoch < 30:
                rate = 1
            elif epoch < 60:
                rate = 0.1
            elif epoch < 80:
                rate = 0.01
            else:
                rate = 0.001
            return rate * lr
        trainer.extend(warmup_and_exponential_shift)

        evaluator = chainermn.create_multi_node_evaluator(
            extensions.Evaluator(val_iter, model, device=device), comm)
        trainer.extend(evaluator, trigger=(1, 'epoch'))
        trainer.extend(comm.get_uninitializer(), trigger=(1, 'iteration'))

        log_interval = 0.1, 'epoch'
        print_interval = 0.5, 'epoch'
        plot_interval = 1, 'epoch'

        if comm.intra_rank == 0:
            # TODO: lr is not properly controlled for accuracy
            trainer.extend(chainer.training.extensions.observe_lr(),
                           trigger=log_interval)
            trainer.extend(echainer.extension.Lineage(comm, trigger=log_interval))
            trainer.extend(extensions.PrintReport(
                ['iteration', 'epoch', 'elapsed_time', 'lr',
                 'main/loss', 'validation/main/loss',
                 'main/accuracy', 'validation/main/accuracy'],
                log_report='Lineage'), trigger=print_interval)
            trainer.extend(extensions.ProgressBar(update_interval=10))

            if extensions.PlotReport.available():
                trainer.extend(
                    extensions.PlotReport(
                        ['main/loss', 'validation/main/loss'],
                        file_name='loss.png', trigger=plot_interval
                    ),
                    trigger=plot_interval
                )
                trainer.extend(
                    extensions.PlotReport(
                        ['main/accuracy', 'validation/main/accuracy'],
                        file_name='accuracy.png', trigger=plot_interval
                    ),
                trigger=plot_interval
                )

        # Optimizer includes model parameters and other params in optimizer
        comm.register_state('optimizer', optimizer)
        comm.register_state('iterator', train_iter)
        if retry or not comm.initial:
            (iteration, epoch) = comm.fetch_state('optimizer', optimizer)
            # train_iter.epoch = epoch
            comm.fetch_state('iterator', train_iter)
            updater.iteration = iteration

        optimizers = trainer.updater.get_all_optimizers()
        for name in optimizers.keys():
            optimizers[name].reset_prev_params()

        try:
            print('start trainer.run(), ', trainer.updater.iteration, trainer.updater.epoch)
            trainer.run()
            done = trainer._done
        except CommException as ce:
            print("Comm exception >>>>>>>>>>>", ce, updater.iteration, updater.epoch)
            comm.save_all_states(updater.iteration, updater.epoch)
            # Here comm will be ready to accept fetch state calls and once all
            # nodes got catched up it'll return and continue to run: TODO
            comm.sync_cluster(trainer.updater.get_all_optimizers())
            retry = True
            continue
        except ClusterUpdatedException as ce:
            print("Cluster updated: >>>>>>>>>>>", ce)
            comm.save_all_states(updater.iteration, updater.epoch)
            comm.sync_cluster(trainer.updater.get_all_optimizers())
            retry = True
            continue
        except Exception as e:
            print("Unexpected >>>>>>>>>>>", e)
            break

    comm.leave()


if __name__ == '__main__':
    main()
