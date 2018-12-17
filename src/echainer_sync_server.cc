#include "echainer_sync_server.h"

#include <thread>
#include <glog/logging.h>

namespace echainer {

  EchainerSyncServer::EchainerSyncServer(const std::string& host):
    Echainer::Service(), has_assign_(false), reset_count_(0), host_(host) {
  }
  EchainerSyncServer::EchainerSyncServer(): Echainer::Service() {
  }
  EchainerSyncServer::~EchainerSyncServer() {
    // TODO: this is VERY BAD workaround; deal with it.
    LOG(INFO) << "Shutdown gRPC server";
    // std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  // MPI-like gRPC APIs
  Status EchainerSyncServer::AssignRank(ServerContext* context,
                                      const AssignRankRequest* req, AssignRankResponse* res) {
    // Broadcast data pushed from root
    DLOG(INFO) << "AssignRank from " << context->peer() << " ~ " << req->rank() << " hogehoge";
    std::unique_lock<std::mutex> lk(m_);
    assign_ = *req;
    has_assign_ = true;
    cv_.notify_one();
    //DLOG(INFO) << "Broadcast recv ok";
    return Status::OK;
  }
  // receive a data
  Status EchainerSyncServer::SendBlock(ServerContext* context, const SendBlockRequest* req, SendBlockResponse*) {
    DLOG(INFO) << "Sendblock from " << context->peer() << "/" << req->from() << " a blog of "
               << req->block().size() << " bytes";
    std::unique_lock<std::mutex> lk(m_);
    // TODO: remove duplicate message
    recv_buffer_.push_back(*req);
    cv_.notify_one();
    return Status::OK;
  }
  Status EchainerSyncServer::FetchState(ServerContext* context,
                                      const FetchStateRequest* req,
                                      FetchStateResponse* res) {
    DLOG(INFO) << "FetchState from " << context->peer() << "/" << req->from() << " on "
              << req->name();
    const std::string name = req->name();
    std::unique_lock<std::mutex> lk(m_);

    if (registered_state_.find(name) == registered_state_.end()) {
      LOG(ERROR) << "No state for name " << name << " found.";
      return grpc::Status(grpc::StatusCode::NOT_FOUND, "Not found");
    }
    res->set_iteration(iteration_);
    res->set_epoch(epoch_);
    auto state = registered_state_[name];
    size_t b = req->offset();
    size_t e = std::min(b + 4000000, state.size()); // 4MB hard code
    std::string substr = state.substr(b, e-b);
    res->set_state(substr);
    res->set_total(state.size());
    DLOG(INFO) << "Responding " << res->state().size() << " bytes of total=" << res->total()
              << " b=" << b << " e=" << e;
    return grpc::Status::OK;
  }

  Status EchainerSyncServer::Recv(size_t src, std::string& ret) {
    size_t retry = 16; // TODO: hardcoded
    while (retry-- > 0) {
      std::unique_lock<std::mutex> lk(m_);
      for (size_t i = 0; i < recv_buffer_.size(); i++) {
        SendBlockRequest& req = recv_buffer_[i];
        if (req.from() == src) {
          ret = req.block();
          recv_buffer_.erase(recv_buffer_.begin() + i);
          return grpc::Status::OK;
        }
      }
      // Not found. block for a while.
      cv_.wait_for(lk, std::chrono::milliseconds(500));
    }
    return grpc::Status(grpc::StatusCode::CANCELLED, "Retry time over");
  }
  void EchainerSyncServer::ResetBuffer() {
    std::unique_lock<std::mutex> lk(m_);
    recv_buffer_.clear();
    reset_count_++;
  }
  unsigned long EchainerSyncServer::ResetCount() const {
    return reset_count_;
  }

  bool EchainerSyncServer::Listen() noexcept {
    grpc::ServerBuilder builder;

    //Prohibit reuseport to guarantee process uniqueness
    //grpc::ChannelArguments args;
    //args.SetInt(GRPC_ARG_ALLOW_REUSEPORT, 0);
    // => inherit grpc::ServerBuildOption and instanciat as opt;
    // => builder.SetOption(opt);

    // http://nanxiao.me/en/message-length-setting-in-grpc/
    builder.SetMaxReceiveMessageSize(INT_MAX);
    builder.AddListeningPort(host_, grpc::InsecureServerCredentials());
    builder.RegisterService(this);
    std::shared_ptr<grpc::Server> server(builder.BuildAndStart());
    if (server.get() == nullptr) {
      return false;
    }
    server_ = server;
    LOG(INFO) << "gRPC Server started: " << host_;
    return true;
  }

  void EchainerSyncServer::Shutdown() noexcept {
    server_->Shutdown();
  }

  bool EchainerSyncServer::WaitForAssign(AssignRankRequest& assign, unsigned int timeout_ms) {
    ssize_t count = 5;
    auto timeout = std::chrono::milliseconds(timeout_ms / count);
    do {
      std::unique_lock<std::mutex> lk(m_);
      if (has_assign_) {
        assign = assign_;
        has_assign_ = false;
        return true;
      }
      //auto now = std::chrono::system_clock::now();
      //auto until = now + std::chrono::duration_cast<std::chrono::millisecond>(timeout_ms);
      cv_.wait_for(lk, timeout); // TODO: use wait_until for timeouts
    } while (count-- > 0);

    return false;
  }

  bool EchainerSyncServer::RegisterState(size_t iteration, size_t epoch,
                                       const std::string& name,
                                       const char* ptr, size_t len) {
    std::unique_lock<std::mutex> lk(m_);
    iteration_ = iteration;
    epoch_ = epoch;
    registered_state_.insert(std::pair<std::string, std::string>(name, std::string(ptr, len)));
    return true;
  }

}
