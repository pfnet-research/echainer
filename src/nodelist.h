// Copyright 2018 Preferred Networks, Inc.
#pragma once

#include <vector>
#include <set>
#include <string>
#include <map>

namespace echainer {
  /// Verify the list is correctly set,
  //  - continous processes in same list
  //  - no duplicate nodes or processes
  //  -
  bool verify_nodelist(const std::vector<std::string>& list, bool initial);
  void sort_nodelist(std::vector<std::string>& list);
  size_t calc_ppn(const std::vector<std::string>& list);

  void aggregate_ports(const std::vector<std::string>& list,
                      std::vector<std::pair<std::string, std::vector<unsigned int>>>& sorted_list);
  void nodelist2signature(const std::map<std::string, std::string>& list,
                          std::string& signature);
}
