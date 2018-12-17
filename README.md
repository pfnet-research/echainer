# Elastic Chainer

Scalable and fault tolerant MPI-like communication library prototype
for distributed DNN training in Chainer.

## Build

### Prerequisites

Runtime

- [protobuf](https://developers.google.com/protocol-buffers/) - to run gRPC server
- [grpc](https://grpc.io/) - for processes to commumnicate each other
- [cppetcd](https://github.com/kuenishi/cppetcd) - to access etcd for distributed coordination
- [glog](https://github.com/google/glog) - for simple logging


Build

- `protoc` - protobuf compiler to generate codes
- `g++` - C++ compiler and linkers that support C++14.
- `pkg-config` - find all libraries above on compile time

### Build (TL;DR)

You'll need `python`, `protoc`, and `grpc++` for build. Also `cppetcd` as well.

```
$ yay -S python protobuf grpc glog
$ which protoc
/usr/bin/protoc
$ which grpc_cpp_plugin
/usr/bin/grpc_cpp_plugin
$ git clone git@github.com:kuenishi/cppetcd.git
$ cd cppetcd && make && make install prefix=/path/to/local && cd ..
$ git clone git@github.com:chainer/echainer.git
$ cd echainer
$ make dev
```

For build it doesn't need CuPy, but in runtime `NcclCommunicator` requires it. Also see `docker/Dockerfile` on how to build cleanly.

### Release Package

Build a release binary package (be sure only single platform will be supported by this single build).

```
$ pip install wheel
$ python setup.py bdist_wheel
```
## Development

Running [etcd](https://coreos.com/etcd/) process in `localhost:2379` to run `make test` is required. To run GPU-less test:

```
$ make test
```

To run tests with GPU,

```
$ make gpu-test
```

## Loglevel

`GLOG_logtostderr=1`

## Copyright

(C) 2018 Preferred Networks, Inc.
