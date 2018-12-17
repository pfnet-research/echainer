#pragma once
#include <vector>
#include <string>
#include <cppetcd.h>
#include <thread>
#include <atomic>

namespace echainer {
  class FailureDetector : public etcd::EventWatcher {
  public:
    FailureDetector(std::shared_ptr<etcd::Client>& c,
                    const std::string& prefix,
                    const std::vector<std::string>& hosts);
    virtual ~FailureDetector();

    virtual void HandleEvents(const std::vector<etcd::KeyValueEvent>& events);
    virtual bool StopHandling() const;
    
    void Start();
    unsigned int GetGen() const;

  private:
    std::shared_ptr<etcd::Client>& c_;
    std::string prefix_;
    std::vector<std::string> hosts_;
    std::thread watcher_;
    std::atomic_uint generation_; // view generation;
    volatile bool live_;
  };
}
