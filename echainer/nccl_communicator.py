import json
import numpy as np

import chainer
from echainer import _memory_utility

from echainer.communicator import MetaCommunicator, CommunicatorResource
from echainer_internal import CommException

class NcclResource(CommunicatorResource):
    def __init__(self):
        super(NcclResource, self).__init__()
        self.nccl_comm = None

    def make_sesame(self, intra_rank):
        print('Using GPU No.', intra_rank, " @to get unique id")
        from cupy.cuda import nccl
        chainer.cuda.get_device_from_id(intra_rank).use()
        uid = nccl.get_unique_id()
        return json.dumps(uid)

    def construct(self, size, sesame, rank):
        from cupy.cuda import nccl
        comm_id = tuple(json.loads(sesame))
        self.nccl_comm = nccl.NcclCommunicator(size, comm_id, rank)
        print('NCCL initialized:', size, rank)
        assert self.nccl_comm is not None

    def destroy(self):
        if hasattr(self, 'nccl_comm') and self.nccl_comm is not None:
            self.nccl_comm.destroy()
            self.nccl_comm = None


class NcclCommunicator(MetaCommunicator):
    def __init__(self, policy, etcd='etcd://localhost:2379/hogehoge',
                 bind='localhost:20001'):
        self.nccl_resource = NcclResource()
        super(NcclCommunicator, self).__init__(policy, etcd, bind,
                                               resources=[self.nccl_resource])

        self.gpu_tmp_buffer = _memory_utility.DeviceMemory()
        self.gpu_buffer_a = _memory_utility.DeviceMemory()
        self.gpu_buffer_b = _memory_utility.DeviceMemory()

        self.allreduce_grad_dtype = None
        self.grad_dtype_to_allreduce_dtype_kernel = None
        self.allreduce_dtype_to_grad_dtype_kernel = None
        self.div_by_size = None

    def bcast_data(self, model):
        params = _memory_utility.extract_params_set_data(model)
        data_dtype = _get_param_data_dtype(params[0])
        n_elems = sum(param.data.size for param in params)
        data_grad_n_bytes = data_dtype.itemsize * n_elems
        if self.gpu_tmp_buffer.size != data_grad_n_bytes:
            self.gpu_tmp_buffer.assign(data_grad_n_bytes)

        stream = chainer.cuda.Stream.null

        _memory_utility.pack_params(
            params, data_dtype.itemsize, 'data',
            self.gpu_tmp_buffer, stream)
        print("Broadcastin'", data_dtype, self.gpu_tmp_buffer.ptr(), n_elems, self.rank)

        nccl_comm = self.nccl_resource.nccl_comm
        if nccl_comm is None:
            raise CommException()
        nccl_comm.bcast(self.gpu_tmp_buffer.ptr(), n_elems,
                        _get_nccl_type_id(data_dtype), 0, stream.ptr)
        print("bcast done", self.rank)

        _memory_utility.unpack_params(
            params, data_dtype.itemsize, 'data',
            self.gpu_tmp_buffer, stream)

    def broadcast_data(self, model):
        self.bcast_data(model)

    def allreduce_grad(self, model):
        stream = chainer.cuda.Stream.null
        self._allreduce_grad_async(model, stream)

    def _allreduce_grad_async(self, model, stream):
        from cupy.cuda import nccl
        params = _memory_utility.extract_params_set_grad(model)
        grad_dtype = _get_param_grad_dtype(params[0])
        if self.allreduce_grad_dtype is None:
            allreduce_grad_dtype = grad_dtype
        else:
            allreduce_grad_dtype = self.allreduce_grad_dtype
        n_elems = sum(param.grad.size for param in params)
        needs_sync = self._assign_for_allreduce_grad(grad_dtype,
                                                     allreduce_grad_dtype,
                                                     n_elems)
        if stream != chainer.cuda.Stream.null and needs_sync:
            chainer.cuda.Stream.null.synchronize()

        self._pack_params_to_buffer(params, grad_dtype, allreduce_grad_dtype,
                                    n_elems, stream)

        nccl_comm = self.nccl_resource.nccl_comm
        if nccl_comm is None:
            raise CommException()
        #print("allreduce in")
        nccl_comm.allReduce(self.gpu_buffer_a.ptr(),
                            self.gpu_buffer_b.ptr(), n_elems,
                            _get_nccl_type_id(allreduce_grad_dtype),
                            nccl.NCCL_SUM,
                            stream.ptr)
        #print("allreduce out")
        if self.div_by_size is None:
            self.div_by_size = chainer.cuda.cupy.ElementwiseKernel(
                '{} x'.format(allreduce_grad_dtype.name),
                '{} y'.format(allreduce_grad_dtype.name),
                'y = x*(1.0/{})'.format(self.size), 'div_by_size')
        self.div_by_size(
            self.gpu_buffer_b.array(n_elems,
                                    dtype=allreduce_grad_dtype),
            self.gpu_buffer_a.array(n_elems,
                                    dtype=allreduce_grad_dtype),
            stream=stream)
        self._unpack_params_from_buffer(params, grad_dtype,
                                        allreduce_grad_dtype, n_elems, stream)

    def _assign_for_allreduce_grad(self, grad_dtype, allreduce_grad_dtype,
                                   n_elems):
        allreduce_grad_n_bytes = allreduce_grad_dtype.itemsize * n_elems
        needs_sync = False
        if self.gpu_buffer_a.size != allreduce_grad_n_bytes:
            self.gpu_buffer_a.assign(allreduce_grad_n_bytes)
            needs_sync = True
        if self.gpu_buffer_b.size != allreduce_grad_n_bytes:
            self.gpu_buffer_b.assign(allreduce_grad_n_bytes)
            needs_sync = True

        if grad_dtype != allreduce_grad_dtype:
            grad_n_bytes = grad_dtype.itemsize * n_elems
            if self.gpu_tmp_buffer.size != grad_n_bytes:
                self.gpu_tmp_buffer.assign(grad_n_bytes)
                needs_sync = True
        return needs_sync

    def _pack_params_to_buffer(self, params, grad_dtype, allreduce_grad_dtype,
                               n_elems, stream):
        if grad_dtype == allreduce_grad_dtype:
            _memory_utility.pack_params(
                params, grad_dtype.itemsize, 'grad',
                self.gpu_buffer_a, stream=stream)
        else:
            if self.grad_dtype_to_allreduce_dtype_kernel is None:
                self.grad_dtype_to_allreduce_dtype_kernel = \
                    _get_converting_kernel(
                        grad_dtype, allreduce_grad_dtype,
                        'grad_dtype_to_allreduce_dtype_kernel')

            _memory_utility.pack_params(
                params, grad_dtype.itemsize, 'grad',
                self.gpu_tmp_buffer, stream=stream)

            self.grad_dtype_to_allreduce_dtype_kernel(
                self.gpu_tmp_buffer.array(n_elems, dtype=grad_dtype),
                self.gpu_buffer_a.array(n_elems,
                                        dtype=allreduce_grad_dtype),
                stream=stream)

    def _unpack_params_from_buffer(self, params, grad_dtype,
                                   allreduce_grad_dtype, n_elems, stream):
        if grad_dtype == allreduce_grad_dtype:
            _memory_utility.unpack_params(
                params, allreduce_grad_dtype.itemsize, 'grad',
                self.gpu_buffer_a, stream)

        else:
            if self.allreduce_dtype_to_grad_dtype_kernel is None:
                self.allreduce_dtype_to_grad_dtype_kernel = \
                    _get_converting_kernel(
                        allreduce_grad_dtype, grad_dtype,
                        'allreduce_dtype_to_grad_dtype_kernel')
            self.allreduce_dtype_to_grad_dtype_kernel(
                self.gpu_buffer_a.array(n_elems,
                                        dtype=allreduce_grad_dtype),
                self.gpu_tmp_buffer.array(n_elems, dtype=grad_dtype),
                stream=stream)

            _memory_utility.unpack_params(
                params, grad_dtype.itemsize, 'grad', self.gpu_tmp_buffer,
                stream=stream)


def _get_converting_kernel(src_dtype, dst_dtype, kernel_name):
    return chainer.cuda.cupy.ElementwiseKernel(
        '{} x'.format(src_dtype.name),
        '{} y'.format(dst_dtype.name),
        'y = x', kernel_name)


def _get_param_data_dtype(param):
    return param.data.dtype


def _get_param_grad_dtype(param):
    return param.grad.dtype


def _get_nccl_type_id(dtype):
    from cupy.cuda import nccl
    if dtype == np.float16:
        return nccl.NCCL_FLOAT16
    elif dtype == np.float32:
        return nccl.NCCL_FLOAT32
    elif dtype == np.float64:
        return nccl.NCCL_FLOAT64
    else:
        raise ValueError(
            'dtype must be float16, float32, or float64.')
