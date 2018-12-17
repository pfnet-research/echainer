#include "nodelist.h"
#include <functional>
#include <set>
#include <sstream>

#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>

#include <glog/logging.h>

namespace echainer {
  bool verify_nodelist(const std::vector<std::string>& list, bool initial) {
    std::set<std::string> s;

    // Check uniqueness
    for (std::string host : list) {
      s.insert(host);
    }
    if (s.size() != list.size()) {
      LOG(ERROR) << "The list has duplicate entries.";
      return false;
    }

    // verify that all processes are in continous order by host
    s.clear(); // s is known and verified host names
    std::vector<std::pair<std::string, std::vector<unsigned int>>> sorted_list;
    for (std::string host : list) {
      size_t offset = host.find_first_of(":");
      // Check all hosts are in correct format as <hostname>:<port>
      std::string hostname = host.substr(0, offset);
      unsigned int port;
      std::stringstream stream(host.substr(offset+1, hostname.size()));
      stream >> port;

      if (s.find(hostname) == s.end()) {
        /*
         * Doesn't work with raw IP addresses, which will be provided by Kubernetes directly
        struct addrinfo hints =
          {
           .ai_flags = 0,
           .ai_family = AF_INET | AF_INET6,
           .ai_socktype = 0,
           .ai_protocol = 0
          };
        struct addrinfo* result;
        int r = getaddrinfo(hostname.c_str(), NULL, &hints, &result);
        if (r != 0) {
          LOG(ERROR) << "getaddrinfo(3) on " << hostname << " failed: " << gai_strerror(r);
          return false;
          } */
        s.insert(hostname);
        //        freeaddrinfo(result);
      }
      if (port < 1024 or 65535 < port) {
        LOG(ERROR) << "Invalid port number " << port << " for : " << host;
        //LOG(ERROR) << host.substr(offset, hostname.size());
        return false;
      }

      if (sorted_list.size() == 0 or
          sorted_list.back().first != hostname) {
        std::vector<unsigned int> ports;
        ports.push_back(port);
        sorted_list.push_back(std::pair<std::string, std::vector<unsigned int>>(hostname, ports));
      } else {
        sorted_list.back().second.push_back(port);
      } 
    }
    if (s.size() != sorted_list.size()) {
      LOG(ERROR) << "The input list was not correctly sorted:" << s.size() << " != " << sorted_list.size();
      return false;
    }

    if (initial) {
      // Verify all nodes have same number of processes
      unsigned int ppn = sorted_list.front().second.size();
      for (std::pair<std::string, std::vector<unsigned int>> pair : sorted_list) {
        if (pair.second.size() != ppn) {
          LOG(ERROR) << pair.first << " does not have equal process number: " << pair.second.size()
                     << " != " << ppn << " (other nodes' proc number)";
          return false;
        }
      }
    }
    return true;
  }

  void sort_nodelist(std::vector<std::string>& list){
    std::vector<std::pair<std::string, std::vector<unsigned int>>> sorted_list;
    aggregate_ports(list, sorted_list);
    list.clear();
    for (std::pair<std::string, std::vector<unsigned int>> host : sorted_list) {
      for (unsigned int port : host.second) {
        std::stringstream s;
        s << host.first << ":" << port;
        list.push_back(s.str());
      }
    }
  }

  // Assumes the list is already verified
  size_t calc_ppn(const std::vector<std::string>& list){
    std::vector<std::pair<std::string, std::vector<unsigned int>>> sorted_list;
    aggregate_ports(list, sorted_list);
    
    // Verify all nodes have same number of processes
    unsigned int ppn = sorted_list.front().second.size();
    for (std::pair<std::string, std::vector<unsigned int>> pair : sorted_list) {
      if (pair.second.size() != ppn) {
        // Never reaches here
        LOG(FATAL) << pair.first << " does not have equal process number: " << pair.second.size()
                   << " != " << ppn << " (other nodes' proc number)";
      }
    }
    return ppn;
  }

  void aggregate_ports(const std::vector<std::string>& list,
                      std::vector<std::pair<std::string, std::vector<unsigned int>>>& sorted_list){

    for (std::string host : list) {
      size_t offset = host.find_first_of(":");
      // Check all hosts are in correct format as <hostname>:<port>
      std::string hostname = host.substr(0, offset);
      unsigned int port;
      std::stringstream stream(host.substr(offset+1, hostname.size()));
      stream >> port;

      if (sorted_list.size() == 0 or
          sorted_list.back().first != hostname) {
        std::vector<unsigned int> ports;
        ports.push_back(port);
        sorted_list.push_back(std::pair<std::string, std::vector<unsigned int>>(hostname, ports));
      } else {
        sorted_list.back().second.push_back(port);
      } 
    }
  }

  void nodelist2signature(const std::map<std::string, std::string>& list,
                          std::string& signature) {
    std::stringstream ss;
    for (auto pair : list) {
      ss << pair.first << "\t" << pair.second << std::endl;
    }
    std::stringstream ss2;
    ss2 << std::hash<std::string>{}(ss.str());
    signature = ss2.str();
  }

}
