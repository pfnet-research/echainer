#include <thread>
#include <vector>

#include <pybind11/pybind11.h>
// #include <pybind11/stl_bind.h>
#include <pybind11/stl.h>

#include "membership.h"
#include "ticker.h"
#include <stdexcept>

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<std::string>);

PYBIND11_MODULE(echainer_internal, m) {
  m.doc() = "Internal API of Echainer Cluster Manager C++ implementations";
  py::class_<echainer::Membership, std::shared_ptr<echainer::Membership> > membership(m, "Membership");

  // py::register_exception<echainer::CommException>(m, "CommException");
  py::register_exception<std::runtime_error>(m, "CommException");
  
  membership.def(py::init<const std::string&, const std::string&>())
    // .def("get_list", &Membership::GetList, "hehe")
    .def("put_sesame", &echainer::Membership::Put, "put sesame to blackboard")
    .def("get_sesame", &echainer::Membership::Get, "get sasame from blackboard")
    .def("del_sesame", &echainer::Membership::Delete, "delete sasame from blackboard")
    
    .def("send", &echainer::Membership::Send, "Send data")
    .def("recv", &echainer::Membership::Recv, "Recv data")
    .def("bcast", &echainer::Membership::Bcast, "Bcast")
    .def("allreduce", &echainer::Membership::Allreduce, "Allreduce inplace")

    .def("is_initial", &echainer::Membership::Initial, "is initial?")
    .def("set_initial", &echainer::Membership::SetInitial, "set initial")
    .def("get_rank", &echainer::Membership::Rank, "rank")
    .def("get_size", &echainer::Membership::Size, "size")
    .def("get_intra_rank", &echainer::Membership::IntraRank, "intra rank")
    .def("get_intra_size", &echainer::Membership::IntraSize, "intra size")

    .def("set_policy", &echainer::Membership::SetPolicy, "Set ScalePolicy object")
    .def("rendezvous", &echainer::Membership::RendezVous, "Do rendez-vous over etcd/path")
    .def("maybe_update_cluster", &echainer::Membership::MaybeUpdateCluster,
         "Maybe update cluster state by checking gaps")
    .def("should_abort", &echainer::Membership::ShouldAbort,
         "Should be aborted or not")
    .def("fetch_state", &echainer::Membership::FetchState,
         "Fetch remote status for catch up")
    .def("register_state", &echainer::Membership::RegisterState,
         "Register states to sync in recovery")

    .def("abort", &echainer::Membership::Abort, "Abort the whole cluster as much as can")
    .def("leave", &echainer::Membership::Leave, "Gracefully leave or dissolve the cluster");

  py::class_<echainer::ThresholdTicker, std::shared_ptr<echainer::ThresholdTicker> > ticker(m, "ThresholdTicker");

  ticker.def(py::init<>())
    .def("set_int_param", &echainer::ThresholdTicker::SetIntParam, "hehe")
    .def("get_int_param", &echainer::ThresholdTicker::GetIntParam, "put sesame to blackboard")
    .def("is_alive", &echainer::ThresholdTicker::Alive, "get sasame from blackboard")
    .def("tick", &echainer::ThresholdTicker::Tick, "delete sasame from blackboard");

}
