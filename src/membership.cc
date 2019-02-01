#include <functional>
#include <sstream>
#include <thread>
#include <vector>

#include <pybind11/pybind11.h>
#include <glog/logging.h>
#include "membership.h"

#include <cppetcd.h>

#include "nodelist.h"

namespace py = pybind11;

namespace echainer {

  Membership::Membership(const std::string& addr, const std::string& prefix):
    //addr_(addr), lock_name_("etcd://192.176.23.32/doggo-doggo-doggo"), prefix_(prefix), constructed_(false)
    addr_(addr), prefix_(prefix), constructed_(false), view_hash_(""), initial_(true)
    //state(WAIT)
  {
    // Using lock name as hashed value, as etcd seems NOT accepting
    // lock name like 'etcd://<ipaddr>:2379/doggo-doggo' or already
    // acquired by some etcd builtin system? Thus we can't use raw
    // etcd prefix.
    size_t h = std::hash<std::string>{}(prefix);
    std::stringstream str;
    str << h;
    this->lock_name_ = str.str(); // wanna make this as const variable
    rpc_server_ = std::make_shared<echainer::EchainerSyncServer>(addr_);
  }
  Membership::~Membership() {
    if (etcd_client_->Connected()) {
      etcd_client_->Disconnect();
    }
    kapoller_.join();
    DLOG(INFO) << "destructor~";
  }

  // TBD: should this be done within constructor?
  // Start rendez-vous here
  // - Connect to etcd/path
  // - Get process list in the prefix
  // - Try to take a lock if you're 1st in the list
  // - Build provisional process list by assigning ranks
  // - callback ok2run()
  //    - 'wait': wait
  //    - 'fail': fail return
  //    - ('ok', True):
  //       - notify all processes with rank assign
  //       - initialize NCCL2
  //       - in all process callback to reconstruct()
  //    - ('ok', False): do nothing
  // - all done, rendez-vous finish
  bool Membership::RendezVous(){
    // Start rendez-vous here
    // Release the GIL before blocking call
    // http://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil
    //py::gil_scoped_release release;
    // - Connect to etcd/path
    grpc::Status status = connect_etcd_();
    if (not status.ok()) {
      LOG(ERROR) << "Cannot connect to etcd: " << status.error_message();
      return false;
    }
    if (not rpc_server_->Listen()) {
      LOG(FATAL) << "Failed to start RPC Server @ " << addr_;
    }

    std::string prefix = prefix_ + "/hosts/";
    update_list_();

    failure_detector_ = std::make_shared<FailureDetector>(etcd_client_, prefix, hosts_);
    failure_detector_->Start();
    generation_ = failure_detector_->GetGen();

    update_cluster_();
    generation_ = failure_detector_->GetGen();
    LOG(INFO) << "Cluster starting with " << hosts_.size() << "hosts";

    return true;
    // - all done, rendez-vous finish
  }

  void Membership::SetPolicy(py::object& policy) {
    policy_ = policy;
  }

  bool Membership::MaybeUpdateCluster() { // returns ok to run or not
    unsigned int remote_generation = failure_detector_->GetGen();
    //LOG(INFO) << "MaybeDestroy:" << remote_generation << " <> " << generation_
    //<< "  constructed_=" << constructed_ << " initial_=" << initial_;
    if (remote_generation == generation_ && constructed_) {
      // No updates, nothing to do
      return false;
    }

    // state = STALE;
    // something is updated; refresh!
    update_cluster_();
    generation_ = failure_detector_->GetGen();
    return true;
  }

  void Membership::update_cluster_() {
    bool initial = initial_;
    do {
      try_lock_();
      // - Build provisional process list by assigning ranks
      update_list_();

      if (etcd_client_->HasLock(lock_name_)) {
        LOG(INFO) << "Now I have lock! I'm master! by " << addr_;
        //    - 'wait': wait
        // TODO: sleep a while
        //    - 'fail': fail return
        // On initial bootstrap it doesn't fail, just wait
        //    - 'ok': use the list as it is
        //       - notify all processes with rank assign
        //       - initialize NCCL2
        //       - in all process callback to reconstruct()
        //    - ('ok', list): use the returned list to init
        //       - notify all processes with rank assign
        //       - initialize NCCL2
        //       - in all process callback to reconstruct()

        switch( eval_ok2run_(initial) ) {
        case WAIT: {
          ::sleep(3);
            continue;
        }
        case FAIL:
          // TODO: send abort message across the cluster
          if (not initial) {
            LOG(FATAL) << "Failed to form a cluster denied by the policy";
          } else {
            ::sleep(5);
            continue;
          }

        case OK:
          //state = STALE;
          if (initial) {
            if (not verify_nodelist(hosts_, true)) {
              // TODO: Add abort code
              LOG(FATAL) << "Cannot verify node list";
            }
          }
          if (initial or not constructed_) {
            grpc::Status status = form_cluster_();
            if (not status.ok()) {
              LOG(ERROR) << "Cannot form a cluster: " << status.error_message();
              continue;
            }
          }
          return;
        default:
          LOG(FATAL) << "><";
        }
      } else {
        // No lock acquired, wait for locker to assign ranks
        LOG(INFO) << "I don't have lock (T T " << lock_name_ << " ) waiting for Assign. " << addr_;
        /// Wait for rank assign by lock acquiror
        echainer::AssignRankRequest req;
        if (rpc_server_->WaitForAssign(req, 500)) {
          rank_ = req.rank();
          intra_rank_ = req.intra_rank();
          intra_size_ = req.intra_size();
          initial_ = req.initial();

          LOG(INFO) << "New rank and intra_rank assigned: " << rank_ << ", " << intra_rank_;
          //       - in all process callback to reconstruct()
          update_list_();

          if (view_hash_ != req.signature()) {
            // TODO: reject if it has different view?
            LOG(WARNING) << "rank(" << rank_ << "): view_hash=" << view_hash_ << ", master's hash=" << req.signature();
          }

          call_construct_();
          return;
        }
        // else continue;
      }
      DLOG(INFO) << "little progress, retrying...";
    }while (true);
    LOG(FATAL) << "><";
  }

  // Should be called in different thread than main one... hopefully?
  bool Membership::ShouldAbort() {
    unsigned int remote_generation = failure_detector_->GetGen();
    //LOG(INFO) << "MaybeDestroy:" << remote_generation << " <> " << generation_
    //	      << "  constructed_=" << constructed_;
    if (remote_generation == generation_ && constructed_) {
      // No updates, nothing to do
      return false;
    }
    if (not constructed_) {
      return false;
    }
    LOG(INFO) << "Going to destroy(), remote_generation=" << remote_generation
              << " generation=" << generation_ << " constructed=" << constructed_;
    rpc_server_->ResetBuffer();
    //py::object destroy = policy_.attr("destroy");
    //destroy();
    constructed_ = false;
    return true;
  }

  bool Membership::Put(const std::string& name, py::buffer b) {
    std::string key = prefix_ + "/data-" + name;
    py::buffer_info info = b.request();
    std::string payload = std::string(static_cast<const char*>(info.ptr),
                                      info.size * info.itemsize);

    // LOG(INFO) << "Saving " << payload.size() << " bytes as " << key;
    // TODO: delete all data when leaving the cluster and provide a tool to clean up them
    Status status = etcd_client_->Put(key, payload, 0, true);
    if (not status.ok()) {
      LOG(ERROR) << "Cannot put sesame: " << key << ": " << status.error_message();
    }
    return status.ok();
  }

  py::bytes Membership::Get(const std::string& name) {
    std::string sesame;
    std::string key = prefix_ + "/data-" + name;
    long long int rev;
    Status status = etcd_client_->Get(key, sesame, &rev);
    if (not status.ok()) {
      if (status.error_code() != grpc::StatusCode::NOT_FOUND) {
        LOG(ERROR) << "Cannot get sesame " << key << ": " << status.error_message();
      }
      return py::none();
    }
    return py::bytes(sesame);
  }

  bool Membership::Delete(const std::string& name) {
    std::string key = prefix_ + "/data-" + name;
    Status status = etcd_client_->Delete(key, 0);
    if (not status.ok()) {
      LOG(ERROR) << "Cannot delete sesame: " << key << ": " << status.error_message();
    }
    return status.ok();
  }

  bool Membership::RegisterState(size_t iteration, size_t epoch,
                                 const std::string& name, py::buffer buf) {
    py::buffer_info info = buf.request();
    return rpc_server_->RegisterState(iteration, epoch, name,
                                      static_cast<const char*>(info.ptr),
                                      info.size * info.itemsize);
  }

  py::tuple Membership::FetchState(const std::string& name) {
    for (size_t i = 0; i < hosts_.size(); i++) {
      size_t j = (i + rank_ + 1) % hosts_.size(); // next node in rank+i
      if (j == rank_) {
        continue;
      }
      std::string host = hosts_[j];
      // TODO: for host == addr_ avoid fetching via TCP/IP but get from local buffer
      DLOG(INFO) << "Fetching state " << name << " from " << host;

      std::shared_ptr<grpc::ChannelInterface> channel_ = grpc::CreateChannel(host, grpc::InsecureChannelCredentials());
      std::unique_ptr<echainer::Echainer::Stub> stub = echainer::Echainer::NewStub(channel_);
      echainer::FetchStateRequest req;
      req.set_to(0);
      req.set_from(rank_);
      req.set_name(name);

      std::string ret;
      size_t offset = 0;
      do {
	req.set_offset(offset);
	DLOG(INFO) << "here fetching offset=" << offset;
	grpc::ClientContext context;
	// TODO: add timeout
	echainer::FetchStateResponse res;
	// TODO: make this stream
	grpc::Status status = stub->FetchState(&context, req, &res);
	if (not status.ok()) {
	  LOG(ERROR) << "Failed to fetch from " << host << ": " << status.error_message();
	  if (status.error_code() == grpc::StatusCode::NOT_FOUND) {
	    break; // next host
	  }
	  throw std::runtime_error("comm error during fetching state");
	}

	ret.append(res.state());
	offset = ret.size();
	if (res.total() <= ret.size()) {
	  LOG(INFO) << "here fetch done: total=" << res.total() << " got=" << ret.size();
	  return py::make_tuple(res.iteration(), res.epoch(), py::bytes(ret));
	}
      } while(true);
    }
    LOG(FATAL) << "Failed to fetch data from any host: no node has data on " << name;
    return py::none();
  }

  // TODO: copy happens
  void Membership::Send(size_t dest, const std::string& bytes) {
    py::gil_scoped_release release;
    if (dest >= hosts_.size() or dest == rank_) {
      LOG(ERROR) << "Invalid rank: " << dest;
      return;
    }

    // LOG(INFO) << "Sending " << bytes.size() << " bytes to " << dest;
    std::string host = hosts_[dest];

    std::shared_ptr<grpc::ChannelInterface> channel_ = grpc::CreateChannel(host, grpc::InsecureChannelCredentials());
    std::unique_ptr<echainer::Echainer::Stub> stub = echainer::Echainer::NewStub(channel_);
    grpc::ClientContext context;
    echainer::SendBlockRequest req;
    req.set_to(dest);
    req.set_from(rank_);
    req.set_block(bytes);
    echainer::SendBlockResponse res;

    grpc::Status status = stub->SendBlock(&context, req, &res);
    if (not status.ok()) {
      // TODO: retry until success, or failed to verify destination liveness.
      // Once it's liveness disqualified, raise an CommException.
      LOG(ERROR) << "Failed to send to " << host << ": " << status.error_message();
    } else {
      // LOG(INFO) << "Sent " << bytes.size() << " bytes to process " << rank_;
    }
    return;
  }
  py::bytes Membership::Recv(size_t src) {
    if (src >= hosts_.size() or src == rank_) {
      LOG(ERROR) << "Invalid source rank: " << src;
      //return py::none();
      throw std::runtime_error("Invalid rank");
      // TODO: workaround, just ignoring old message may
    }

    unsigned long long rc = rpc_server_->ResetCount();
    std::string ret;
    py::gil_scoped_release release;
    grpc::Status status;
    while (rc == rpc_server_->ResetCount()) {
      status = rpc_server_->Recv(src, ret);
      if (status.ok()) {
	return py::bytes(ret);
      }
      // LOG(INFO) << "Recv end: received bytes=" << ret.size();
    }
    LOG(ERROR) << "Failed to receive data from " << src << ": " << status.error_message();
    throw std::runtime_error("Can't receive message");

  }

  void Membership::Bcast(py::buffer b, size_t root) {
    // LOG(INFO) << "Bcast start";
    py::buffer_info info = b.request();
    if (rank_ == root) {
      std::string payload = std::string(static_cast<const char*>(info.ptr),
                                        info.size * info.itemsize);

      for (size_t s = 0; s < hosts_.size(); s++) {
        if (s != rank_) {
          Send(s, payload);
        }
      }

    } else {
      DLOG(INFO) << "bcast non-root> root=" << root << " rank=" << rank_;
      py::object o = Recv(root);
      if (o.is(py::none())) {
          throw std::runtime_error("communication error during bcast");
      }
      std::string buf = py::cast<std::string>(o);
      if (info.size * info.itemsize != buf.size()) {
        LOG(FATAL) << "Different size received at Bcast: " << info.size * info.itemsize << " <> " << buf.size();
      }
      memcpy(info.ptr, buf.c_str(), buf.size());
    }
    // LOG(INFO) << "Bcast end";
    return;
  }
  void Membership::Allreduce(py::buffer b) {
    size_t next = (rank_ + 1) % hosts_.size();
    size_t prev = (hosts_.size() + rank_ - 1) % hosts_.size();
    // LOG(INFO) << "allreduce start: next=" << next << " prev=" << prev << " size=" << hosts_.size();
    // Verify hosts_ exactly matches current rank_
    py::buffer ret;
    py::buffer_info info = b.request();
    std::string orig = std::string(static_cast<const char*>(info.ptr),
                                   info.size * info.itemsize);
    if (info.itemsize != sizeof(float)) {
      LOG(FATAL) << "Wrong data size " << info.itemsize << " != " << sizeof(float);
    }
    float* ws = (float*)malloc(sizeof(float) * info.size);
    for (ssize_t i = 0; i < info.size; i++) {
      ws[i] = 0;
    }
    std::string carry = orig;
    for (size_t s = 0; s < hosts_.size() - 1; s++) {
      Send(next, carry);
      const float* ptr = static_cast<const float*>(static_cast<const void*>(carry.c_str()));
      for (ssize_t i = 0; i < info.size; i++) {
        ws[i] += ptr[i];
      }
      py::object o = Recv(prev);
      if (o.is(py::none())) {
        free(ws);
        throw std::runtime_error("comm error during allreduce");
      }
      carry = py::cast<std::string>(o);
    }
    const float* ptr = static_cast<const float*>(static_cast<const void*>(carry.c_str()));
    for (ssize_t i = 0; i < info.size; i++) {
      ws[i] += ptr[i];
    }

    memcpy(info.ptr, ws, info.size * info.itemsize);
    free(ws);
    // LOG(INFO) << "allreduce start";

  }


  void Membership::Leave() noexcept {
    //rpc_server_->Shutdown();
    failure_detector_->StopHandling();
    if (etcd_client_->Connected()) {
      etcd_client_->Disconnect();
    }
    // LOG(INFO) << "Left the cluster.";
  }

  void Membership::Abort() noexcept {
    LOG(FATAL) << "Not impelmented yet.";
  }

  bool Membership::Initial() const noexcept {
    return initial_;
  }
  void Membership::SetInitial(bool initial) noexcept {
    initial_ = initial;
  }

  size_t Membership::Rank() const noexcept {
    return rank_;
  }
  size_t Membership::Size() const noexcept {
    return hosts_.size();
  }
  size_t Membership::IntraRank() const noexcept {
    return intra_rank_;
  }
  size_t Membership::IntraSize() const noexcept {
    return calc_ppn(hosts_);
  }

  grpc::Status Membership::connect_etcd_() {
    std::string target = prefix_;
    size_t offset = target.find_first_of("://");
    std::string protocol = target.substr(0, offset);
    if( protocol != "etcd" ) { // etcd://host1:2379,host2:2379,host3:2379
      LOG(FATAL) << "Only etcd is supported: protocol=" << protocol;
    }

    std::string hostnames =target.substr(offset + 3, target.size());

    offset = hostnames.find_first_of("/");

    std::stringstream stream(hostnames.substr(0, offset));
    std::vector<std::string> hosts;
    std::string item;
    while (std::getline(stream, item, ',')) {
      std::string host = item.substr(0, item.size());
      hosts.push_back(host);
      DLOG(INFO) << "etcd host: " << host;
    }

    if (hosts.empty()) {
      LOG(FATAL) << "Empty etcd host list.";
    }
    etcd_client_ = std::make_shared<etcd::Client>(etcd::Client(hosts));
    grpc::Status status = etcd_client_->Connect();
    if (not status.ok()) {
      LOG(ERROR) << "Cannot connect to etcd (" << target << "): " << status.error_message();
      return status;
    }
    // Spawn a keepalive thread to keep etcd lease updated every 5
    // secs
    kapoller_ = std::thread( [this]() {
                               // TODO: check thread safety
                               this->etcd_client_->KeepAlive(true);
                             });

    std::string prefix = prefix_ + "/hosts/" + addr_;
    std::string id = my_id(addr_);

    status = etcd_client_->Put(prefix, id, 0, true);
    LOG(INFO) << "Connected to etcd " << prefix_;
    return status;
  }

  grpc::Status Membership::form_cluster_() {
    //       - notify all processes with rank assign
    update_list_();
    std::vector<std::pair<std::string, std::vector<unsigned int>>> sorted_list;
    aggregate_ports(hosts_, sorted_list);

    inter_size_ = sorted_list.size();
    echainer::AssignRankRequest req;

    req.set_gen(generation_);
    req.set_lease_id(etcd_client_->LeaseId());
    req.set_size(hosts_.size());
    req.set_signature(view_hash_);
    req.set_initial(initial_);

    size_t rank = 0;
    for (std::pair<std::string, std::vector<unsigned int>> host : sorted_list) {
      req.set_intra_size(host.second.size());
      size_t intra_rank = 0;

      for (unsigned int port : host.second) {
        req.set_intra_rank(intra_rank);
        req.set_rank(rank);

        std::stringstream s;
        s << host.first << ":" << port;
        std::string host = s.str();

        if (host == addr_) {
          rank_ = rank;
          intra_rank_ = intra_rank;
          LOG(INFO) << "My rank and intra_rank updated: " << rank_ << ", " << intra_rank_;
        } else {
	  grpc::ClientContext context;
	  echainer::AssignRankResponse res;

          std::shared_ptr<grpc::ChannelInterface> channel_ = grpc::CreateChannel(host, grpc::InsecureChannelCredentials());
          std::unique_ptr<echainer::Echainer::Stub> stub = echainer::Echainer::NewStub(channel_);

          grpc::Status status = stub->AssignRank(&context, req, &res);

          if (not status.ok()) {
            LOG(ERROR) << host << " : " << status.error_message();
            return status;
          }

        }

        rank++;
        intra_rank++;
      }
    }
    //       - in all process callback to reconstruct()
    call_construct_();
    return grpc::Status::OK;
  }


  void Membership::try_lock_() {
    grpc::Status status = etcd_client_->Lock(lock_name_, 500);
    if (not status.ok()) {
      if (status.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED) {
        DLOG(INFO) << "Couldn't get lock within " << 500 << "ms. maybe already locked";
      } else {
        LOG(FATAL) << "Couldn't get lock: " << status.error_details();
      }
    }
  }

  // - Get process list in the prefix
  void Membership::update_list_() {
    // fetch Node list prefix
    std::string prefix = prefix_ + "/hosts/";
    // TODO: maybe thius list should maintained by failuredetector as it's receiving watch
    std::vector<std::pair<std::string, std::string>> raw_hosts;
    grpc::Status  status = etcd_client_->List(prefix, raw_hosts);
    if (not status.ok()) {
      LOG(FATAL) << status.error_details();
    }

    // remove etcd://..../ prefix, which could be done within etcd client.
    host_values_.clear();
    hosts_.clear();
    for (auto raw_host : raw_hosts) {
      // hosts_ = raw_hosts;
      std::string host = raw_host.first.substr(prefix.size(), raw_host.first.size());
      host_values_.insert(std::pair<std::string, std::string>(host, raw_host.second));
      hosts_.push_back(host);
    }
    sort_nodelist_by_policy(hosts_);

    std::string prev_hash = view_hash_;
    nodelist2signature(host_values_, view_hash_);
    if (prev_hash != view_hash_) {
      LOG(INFO) << "Node list updated. View hash = " << view_hash_;
    }
  }

  enum Membership::action Membership::eval_ok2run_(bool initial) {

    // - callback ok2run()
    py::object ok2run = policy_.attr("ok2run");
    py::object result = ok2run(hosts_, initial);
    LOG(INFO) << "ok2run(<" << hosts_.size() << " hosts>, " << initial << ") => " << result;
    if (py::isinstance<py::str>(result)) {
      std::string res = py::cast<std::string>(result);

      if  (res == "wait") {
        return Membership::WAIT;
      } else if (res == "fail") {
        return Membership::FAIL;
      } else if (res == "ok") {
        return Membership::OK;
      } else {
        LOG(FATAL) << "Cannot reach here: " << res;
        return Membership::FAIL;
      }
    }
    LOG(FATAL) << "Cannot reach here: " << result << " list?" << py::isinstance<py::list>(result);
    return Membership::FAIL;
  }

  void Membership::sort_nodelist_by_policy(std::vector<std::string>& hosts) {
    // - callback ok2run()
    std::vector<std::string> new_hosts;
    py::object sorter = policy_.attr("sort_node_list");
    if (not sorter.is(py::none())) {
      py::object result = sorter(hosts);
      if (py::isinstance<py::list>(result)) {
        std::list<py::object> pyhosts = py::cast<std::list<py::object>>(result);
        for (py::object pyhost : pyhosts) {
          std::string host = py::cast<std::string>(pyhost);
          new_hosts.push_back(host);
        }
        hosts = new_hosts; // sorted
      } else {
        LOG(WARNING) << "List returned by Policy.sort_node_list(hosts) is not valid. No sort happened.";
      }
    }
  }

  bool Membership::call_construct_() {
    constructed_ = true;
    return constructed_;
  }

  void Membership::wait_for_destroy_() {
    while (constructed_) {
      py::gil_scoped_release release;
      // wait for destroyer thread destroy the cluster here, as this is the place to reform.
      // shouldn't return here until the destruction finishes...
      ::sleep(1);
    }
  }
  std::string get_starttime(pid_t pid) {
    std::stringstream ss;
    ss << "/proc/" << pid << "/stat";

    std::ifstream fin(ss.str().c_str());
    std::string s;
    for (size_t i = 0; i < 22; i++){
      fin >> s;
      if (i == 1) {
        while (s.back() != ')') {
          fin >> s;
        }
      }
    }
    return s;// 22nd element should be the starttime, see proc(5)
  }
  std::string my_id(const std::string& host) {
    // hash(hostname <host>:<port>, pid, /proc/<pid>/stat => starttime)
    std::stringstream ss;
    ss << host << std::endl;
    pid_t pid = ::getpid();
    ss << pid << std::endl;
    ss << get_starttime(pid);

    size_t h = std::hash<std::string>{}(ss.str());
    ss.clear();
    ss << h;
    return ss.str();
  }
}
