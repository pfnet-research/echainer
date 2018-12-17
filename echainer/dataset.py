def scatter_dataset_by_offset(dataset_size, size, root=0,
                              seed=None)
    assert 0 <= root and root < size and dataset_size > 0

    n_total_samples = dataset_size
    n_sub_samples = (n_total_samples + size - 1) // comm.size

    #for i in range(size):
    #    b = n_total_samples * i // size
    #    e = b + n_sub_samples
    scattered = [ (n_total_samples * i // size, b + n_sub_samples)
                  for i in range(size) ]
    return scattered


def scatter_dataset(dataset, comm, root=0, shuffle=False,
                    seed=None, max_buf_len=256 * 1024 * 1024):
    """Scatter the given dataset to the workers in the communicator.

    The dataset of worker 0 (i.e., the worker whose ``comm.rank`` is 0) is
    scattered to all workers. The given dataset of other workers are ignored.
    The dataset is split to sub datasets of almost equal sizes and scattered
    to workers. To create a sub dataset, ``chainer.datasets.SubDataset`` is
    used.

    Args:
        dataset: A dataset (e.g., ``list``, ``numpy.ndarray``,
            ``chainer.datasets.TupleDataset``, ...).
        comm: ChainerMN communicator or MPI4py communicator.
        shuffle (bool): If ``True``, the order of examples is shuffled
            before being scattered.
        root (int): The root process of the scatter operation.
        seed (int): Seed the generator used for the permutation of indexes.
            If an integer being convertible to 32 bit unsigned integers is
            specified, it is guaranteed that each sample
            in the given dataset always belongs to a specific subset.
            If ``None``, the permutation is changed randomly.
        max_buf_len (int): Max buffer size to be used at broadcasting
            binaries. Must not be larger than 2147483647.
    Returns:
        Scattered dataset.
    """

    assert 0 <= root and root < comm.size

    order = None
    if shuffle and dataset is not None:
        n_total_samples = len(dataset)
        order = numpy.random.RandomState(seed).permutation(
            n_total_samples)

    data = None
    if comm.rank == 0:
        data = (dataset, order)

    data = comm.bcast_obj(data, max_buf_len=max_buf_len, root=0)
    assert data is not None
    (dataset, order) = data

    if comm.rank == root:
        mine = None
        n_total_samples = len(dataset)
        n_sub_samples = (n_total_samples + comm.size - 1) // comm.size

        for i in range(comm.size):
            b = n_total_samples * i // comm.size
            e = b + n_sub_samples

            if i == root:
                mine = chainer.datasets.SubDataset(dataset, b, e, order)
            else:
                comm.send_obj((b, e), dest=i)
        assert mine is not None
        return mine

    else:
        data = comm.recv_obj(source=root)
        assert data is not None
        (b, e) = data
        return chainer.datasets.SubDataset(dataset, b, e, order)
