// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../connector_base.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace lmcache {
namespace connector {

// Key encoding constants — must match fs_l2_adapter.py
static constexpr char KEY_SEP = '@';
static constexpr const char* PATH_SLASH_REPLACEMENT = "-SEP-";
static constexpr const char* FILE_EXT = ".data";
static constexpr const char* TMP_EXT = ".tmp";

// Integrity trailer appended to every object file after the payload.
//
// do_single_set writes the payload, then this trailer, fsyncs the file and
// publishes it; do_single_get reads the payload and verifies the trailer
// (magic, payload length, CRC-32C of the payload). A mismatch is reported
// as a failed load for that key, which the caller treats as a miss. Objects
// written before the trailer existed carry none: their file size equals the
// payload length and they are accepted unverified.
struct ObjectTrailer {
  uint32_t magic;
  uint32_t crc32c;
  uint64_t payload_len;
};
static constexpr uint32_t OBJECT_TRAILER_MAGIC = 0x31434D4CU;  // "LMC1"
static_assert(sizeof(ObjectTrailer) == 16, "ObjectTrailer layout");

// CRC-32C (Castagnoli, reflected, init/xorout 0xFFFFFFFF), hardware
// accelerated on SSE4.2 CPUs. The streaming form folds successive slices:
//   state = crc32c_begin(); state = crc32c_update(state, p, n); ...
//   crc = crc32c_finish(state);
inline uint32_t crc32c_begin() { return 0xFFFFFFFFU; }
uint32_t crc32c_update(uint32_t state, const void* data, size_t len);
inline uint32_t crc32c_finish(uint32_t state) { return state ^ 0xFFFFFFFFU; }
uint32_t crc32c(const void* data, size_t len);

// Payloads are read and written in slices of this size so the checksum of
// each slice is computed while the slice is cache-resident.
static constexpr size_t READ_SLICE_BYTES = size_t{1} << 20;

// Per-worker connection state for the FS connector.
// O_DIRECT engages per request only when both the transfer length and the
// caller's buffer address are multiples of the disk block size; anything
// else falls back to buffered I/O.
struct WorkerFSConn {
  std::filesystem::path base_path;
  std::filesystem::path tmp_dir;  // empty if not configured
  bool use_odirect = false;
  size_t disk_block_size = 0;
  // If > 0, trigger filesystem readahead by issuing a small
  // initial read of this many bytes before reading the rest.
  size_t read_ahead_size = 0;
};

class FSConnector : public ConnectorBase<WorkerFSConn> {
 public:
  FSConnector(std::string base_path, int num_workers,
              std::string relative_tmp_dir = "", bool use_odirect = false,
              size_t read_ahead_size = 0);
  ~FSConnector() override;

 protected:
  WorkerFSConn create_connection() override;
  void do_single_get(WorkerFSConn& conn, const std::string& key, void* buf,
                     size_t len, size_t chunk_size) override;
  void do_single_set(WorkerFSConn& conn, const std::string& key,
                     const void* buf, size_t len, size_t chunk_size) override;
  bool do_single_exists(WorkerFSConn& conn, const std::string& key) override;
  bool do_single_delete(WorkerFSConn& conn, const std::string& key) override;

 private:
  // Build the filesystem-safe filename from a serialized key string.
  //
  // Input key (from NativeConnectorL2Adapter._object_key_to_string):
  //   Current unsalted:
  //     "{model}@{kv_rank:08x}@{object_group:x}@{hash.hex()}"
  //   Current salted appends "@{cache_salt}". Legacy three/four-field keys
  //   without object_group_id remain supported.
  //
  // Output filename (matching fs_l2_adapter.py._object_key_to_filename):
  //   Current unsalted:
  //     "{safe_model}@{kv_rank:#010x}@{object_group:x}@{hash.hex()}.data"
  //   Current salted appends "@{cache_salt}" before ".data".
  //
  // Differences from the input: '/' in model becomes '-SEP-', kv_rank
  // gains a '0x' prefix, and '.data' is appended. All fields after kv_rank
  // are preserved because four fields can represent either a legacy salted
  // key or a current unsalted key.
  static std::string key_to_filename(const std::string& key);

  static std::string replace_all(const std::string& str,
                                 const std::string& from,
                                 const std::string& to);

  static void verify_trailer(const std::filesystem::path& file_path,
                             size_t len, uint32_t payload_crc);
  static void append_trailer_and_sync(const std::filesystem::path& tmp_path,
                                      size_t len, uint32_t payload_crc);
  static void sync_directory(const std::filesystem::path& dir);

  std::string base_path_;
  std::string relative_tmp_dir_;
  bool use_odirect_;
  size_t disk_block_size_;
  size_t read_ahead_size_;
};

}  // namespace connector
}  // namespace lmcache
