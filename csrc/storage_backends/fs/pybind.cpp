// SPDX-License-Identifier: Apache-2.0
#include <string>
#include <utility>

#include <pybind11/pybind11.h>
#include "../connector_pybind_utils.h"
#include "connector.h"

namespace py = pybind11;

PYBIND11_MODULE(lmcache_fs, m) {
  py::class_<lmcache::connector::FSConnector>(m, "LMCacheFSClient")
      .def(py::init([](std::string base_path, int num_workers,
                       std::string relative_tmp_dir, bool use_odirect,
                       size_t read_ahead_size, py::object per_op_workers) {
             return new lmcache::connector::FSConnector(
                 std::move(base_path), num_workers, std::move(relative_tmp_dir),
                 use_odirect, read_ahead_size,
                 lmcache::connector::pybind_utils::parse_per_op_workers(
                     per_op_workers));
           }),
           py::arg("base_path"), py::arg("num_workers"),
           py::arg("relative_tmp_dir") = "", py::arg("use_odirect") = false,
           py::arg("read_ahead_size") = 0,
           py::arg("per_op_workers") = py::none())
          LMCACHE_BIND_CONNECTOR_METHODS(lmcache::connector::FSConnector);
}
