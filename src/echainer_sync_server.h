#pragma once
#include <atomic>
#include <grpc++/grpc++.h>

#include "echainer.grpc.pb.h"

namespace echainer {
  using namespace grpc;

  class EchainerSyncServer final : public Echainer::Service {
  public:
    EchainerSyncServer(const std::string&);
    virtual ~EchainerSyncServer();

    Status AssignRank(ServerContext*, const AssignRankRequest*, AssignRankResponse*);
    Status SendBlock(ServerContext*, const SendBlockRequest*, SendBlockResponse*);
    Status FetchState(ServerContext*, const FetchStateRequest*, FetchStateResponse*);

    Status Recv(size_t, std::string&);
    void ResetBuffer();
    unsigned long ResetCount() const;

    bool WaitForAssign(AssignRankRequest& assign, unsigned int timeout_ms);

    bool RegisterState(size_t, size_t,
                       const std::string&, const char*, size_t);
    
    bool Listen() noexcept ;
    void Shutdown() noexcept;
  private:
    EchainerSyncServer();
    std::mutex m_;
    std::condition_variable cv_;

    std::atomic_bool has_assign_;
    echainer::AssignRankRequest assign_;

    std::vector<SendBlockRequest> recv_buffer_;

    std::map<std::string, std::string> registered_state_;
    size_t iteration_;
    size_t epoch_;
    
    std::string host_;
    std::shared_ptr<grpc::Server> server_;
   
    std::atomic_ulong reset_count_;
  };
}
