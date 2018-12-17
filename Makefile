.PHONY= all clean pb test dev docker

PB_SOURCES=src/echainer.grpc.pb.cc src/echainer.pb.cc src/echainer.grpc.pb.h src/echainer.pb.h

## This variables are used for testing in C++. See setup.py for normal build.
CC_SOURCES=$(wildcard src/echainer*.cc)
TEST_SOURCES=$(glob test/*.cc)

CC_OBJECTS=$(CC_SOURCES:.cc=.o)
TEST_OBJECTS=$(TEST_SOURCES:.cc=.o)

INCLUDES=`pkg-config --cflags protobuf grpc++ grpc`
CXXFLAGS += $(INCLUDES) -g -std=c++11

LIBS=`pkg-config --libs protobuf grpc++ grpc`
LDFLAGS=$(LIBS) -lgrpc++_reflection -ldl -g -lglog -lcppetcd

all: pb

dev: pb
	@banner Building eChainer
	CPATH=$(CPATH):/opt/cuda/include:$(HOME)/local/include CC='ccache c++' python3 setup.py develop

pb: $(PB_SOURCES) 

$(PB_SOURCES): echainer.proto
	protoc --grpc_out=src --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` --cpp_out=src echainer.proto

clean:
	-python setup.py clean
	-rm -rvf *.pb.h *.pb.cc
	-rm -f $(CC_OBJECTS) $(TEST_OBJECTS) $(PB_SOURCES)
	-rm -f *.so

## Build and run unit tests
test: dev
	@echo "Test requires etcd started locally."
	etcdctl ls
	@echo "Non-failure path tests"
	./multirunner.py --np 2 -- pytest -m 'not gpu' --size=2 --intra_rank={{intra_rank}}

ETCDIP=$(shell kubectl  get pod -owide | grep echainer-etcd | awk '{print $$6;}')
gpu-test: #dev
	@echo "Test requires etcd started locally. etcd=${ETCDIP}"
	etcdctl --endpoints="http://${ETCDIP}:2379/" get /
	@echo "Non-failure path tests"
	ETCDIP=$(ETCDIP) ./multirunner.py --np 2 -- pytest --size=2 --intra_rank={{intra_rank}}

%.o: %.cc
	$(CXX) $(CXXFLAGS) -I. -I$(HOME)/local/include -Isrc -c -o $@ $<
