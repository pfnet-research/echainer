#pragma once
#include <atomic>
#include <string>

namespace echainer {
  class Ticker {
  public:
    Ticker();
    virtual ~Ticker();
    virtual Ticker& SetIntParam(const std::string& name, int) = 0;
    virtual int GetIntParam(const std::string& name) const = 0;
    virtual Ticker& SetParam(const std::string& name, const std::string&) = 0;
    virtual const std::string& GetParam(const std::string& name) const = 0;
    virtual bool Alive() const noexcept = 0;
    virtual void Tick() = 0;
  };

  class ThresholdTicker: public Ticker {
  public:
    ThresholdTicker();
    virtual ~ThresholdTicker();
    Ticker& SetIntParam(const std::string& name, int);
    int GetIntParam(const std::string& name) const;
    Ticker& SetParam(const std::string& name, const std::string&);
    const std::string& GetParam(const std::string& name) const;
    bool Alive() const noexcept;
    void Tick();

  private:
    unsigned long long timeout_msec_;
    std::atomic_ullong last_updated_;
  };

  unsigned long now_ms();
}
