#include "ticker.h"

#include <time.h>
#include <glog/logging.h>

namespace echainer {
  Ticker::Ticker() {}
  Ticker::~Ticker() {}

  ThresholdTicker::ThresholdTicker():
    Ticker(),
    last_updated_(now_ms()),
    timeout_msec_(16384) // Default 16 seconds
  {}
  ThresholdTicker::~ThresholdTicker() {}
  Ticker& ThresholdTicker::SetIntParam(const std::string& name, int value) {
    if (name == "timeout_ms" and value > 0) {
      timeout_msec_ = value;
    } else {
      LOG(ERROR) << "Invalid configuration: " << name << "=" << value;
    }
    return *this;
  }
  int ThresholdTicker::GetIntParam(const std::string& name) const {
    if (name == "timeout_ms") {
      return timeout_msec_;
    } else {
      LOG(ERROR) << "Invalid configuration: " << name;
      return 0;
    }
  }
  Ticker& ThresholdTicker::SetParam(const std::string& name, const std::string&) {
    // Nothing to do
    return *this;
  }
  const std::string& ThresholdTicker::GetParam(const std::string& name) const {
    // Nothing to do
    return "";
  };
  bool ThresholdTicker::Alive() const noexcept {
    DLOG(INFO) << last_updated_ << " " << now_ms() << " " << timeout_msec_;
    return ((now_ms() - last_updated_) < timeout_msec_);
  }
  void ThresholdTicker::Tick() {
    // IMO it's safe against w-w race.
    if (Alive()) {
      last_updated_ = now_ms();
    } else {
      LOG(ERROR) << "This ticker is no more alive.";
    }
  }

  unsigned long now_ms() {
    struct timespec t;
    int r = ::clock_gettime(CLOCK_BOOTTIME, &t);
    if (r != 0) {
      char buf[1024];
      ::strerror_r(errno, buf, 1024);
      LOG(FATAL) << buf; // heh, it's really fatal and rare if this system call does not work.
      // See errno() and handle them well
      return 0;
    }
    unsigned long now = 0;
    now += (t.tv_sec * 1000);
    now += t.tv_nsec / 1000000;
    return now;
  }

}
