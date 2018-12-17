#pragma once

#include <thread>
#include <vector>
#include <map>

#include <fstream>
#include <stdlib.h>

#include <pybind11/pybind11.h>
// #include <pybind11/stl_bind.h>
#include <pybind11/stl.h>
#include <glog/logging.h>

#include <cppetcd.h>
#include "echainer_sync_server.h"

#include "failure_detector.h"

namespace py = pybind11;

namespace echainer {

  class Membership final {
  public:
    Membership(const std::string& addr, const std::string& prefix);
    ~Membership();

    // Unique id is a value to make consensus
    bool RendezVous();
    void SetPolicy(py::object& policy);
    bool MaybeUpdateCluster();
    bool MaybeDestroy();

    bool Put(const std::string&, py::buffer);
    py::bytes Get(const std::string&);
    bool Delete(const std::string&);

    bool RegisterState(size_t, size_t, const std::string&, py::buffer);
    py::tuple FetchState(const std::string&);

    void Send(size_t dest, const std::string& bytes);
    py::bytes Recv(size_t src);

    void Bcast(py::buffer, size_t root);
    void Allreduce(py::buffer);

    void Leave() noexcept;
    // TODO: implement abort!!! abort them all!!!
    void Abort() noexcept;
    // bool Ping(const std::string& addr) noexcept {};

    bool Initial() const noexcept;
    void SetInitial(bool) noexcept;

    size_t Rank() const noexcept;
    size_t Size() const noexcept;
    size_t IntraRank() const noexcept;
    size_t IntraSize() const noexcept;

  private:
    const std::string addr_;
    std::string lock_name_;
    const std::string prefix_;
    std::string view_hash_;
    std::atomic_bool initial_;

    size_t rank_;
    size_t intra_rank_;
    size_t intra_size_;
    size_t inter_size_;

    std::shared_ptr<etcd::Client> etcd_client_;
    std::shared_ptr<FailureDetector> failure_detector_;
    std::thread kapoller_;

    std::vector<std::string> hosts_;
    std::map<std::string, std::string> host_values_;

    py::object policy_;
    std::shared_ptr<echainer::EchainerSyncServer> rpc_server_;
    //ncclComm_t nccl_comm_;
    unsigned int generation_; // Too early to introduce lineage
    std::atomic_bool constructed_;

    std::map<std::string, py::buffer> registerred_state_;

    grpc::Status connect_etcd_();
    void update_cluster_();
    grpc::Status form_cluster_();

    void try_lock_();
    void update_list_();
    enum action { WAIT, FAIL, OK };
    //enum state { WAIT, STALE, ACTIVE };

    enum action eval_ok2run_(bool);

    void sort_nodelist_by_policy(std::vector<std::string>& hosts);
    bool call_construct_();
    void wait_for_destroy_();
  };

  std::string my_id(const std::string& host);
}
