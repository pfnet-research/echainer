// Copyright 2018 Preferred Networks, Inc.
#include <glog/logging.h>
#include "failure_detector.h"

namespace echainer {
  FailureDetector::FailureDetector(std::shared_ptr<etcd::Client>& c,
                                   const std::string& prefix,
                                   const std::vector<std::string>& hosts):
    etcd::EventWatcher(),
    c_(c), prefix_(prefix), hosts_(hosts), generation_(0), live_(false)
  {
  }
  FailureDetector::~FailureDetector(){
    live_ = false;
    DLOG(INFO) << "stoppoing failure detector";
    watcher_.join();
    DLOG(INFO) << "stopped";
  }

  void FailureDetector::HandleEvents(const std::vector<etcd::KeyValueEvent>& events) {
    // Progress is guaranteed by std::atomic if I understand correctly
    generation_ += events.size();
    LOG(INFO) << "Generation updated: " << events.size() << " events happened.";
  }
  bool FailureDetector::StopHandling() const {
    return not live_;
  }
  
  void FailureDetector::Start() {
    DLOG(INFO) << "Watching " << prefix_;
    if (live_) {
      LOG(WARNING) << "Watching already started";
      return;
    }
    live_ = true;
    watcher_ = std::thread([this]{
                             grpc::Status status = c_->Watch(prefix_, *this);
                             if (status.ok()) {
                               LOG(INFO) << "Watching " << prefix_ << " finished.";
                             } else {
                               LOG(ERROR) << status.error_message();
                             }
                           });
  }

  unsigned int FailureDetector::GetGen() const {
    return generation_;
  }
  
}
