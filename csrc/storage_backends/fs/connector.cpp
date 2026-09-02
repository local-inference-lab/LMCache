// SPDX-License-Identifier: Apache-2.0

#include "connector.h"
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <algorithm>
#if defined(__x86_64__) || defined(__i386__)
#include <nmmintrin.h>
#define LMCACHE_HAVE_SSE42_CRC 1
#endif

namespace {
std::atomic<uint64_t> next_temp_file_id{0};

// Software CRC-32C: reflected polynomial 0x82F63B78, byte table built once.
struct Crc32cTable {
  uint32_t table[256];
  Crc32cTable() {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = (c & 1) ? (0x82F63B78U ^ (c >> 1)) : (c >> 1);
      table[i] = c;
    }
  }
};

uint32_t crc32c_software(uint32_t crc, const uint8_t* p, size_t len) {
  static const Crc32cTable t;
  while (len--) crc = t.table[(crc ^ *p++) & 0xFFU] ^ (crc >> 8);
  return crc;
}

#ifdef LMCACHE_HAVE_SSE42_CRC
__attribute__((target("sse4.2"))) uint32_t crc32c_hardware(uint32_t crc,
                                                           const uint8_t* p,
                                                           size_t len) {
  while (len >= 8) {
    uint64_t v;
    memcpy(&v, p, 8);
    crc = static_cast<uint32_t>(_mm_crc32_u64(crc, v));
    p += 8;
    len -= 8;
  }
  while (len--) crc = _mm_crc32_u8(crc, *p++);
  return crc;
}
#endif

bool crc32c_hardware_available() {
#ifdef LMCACHE_HAVE_SSE42_CRC
  static const bool available = __builtin_cpu_supports("sse4.2");
  return available;
#else
  return false;
#endif
}

// O_DIRECT requires the buffer ADDRESS to be block-aligned as well as the
// length; a misaligned address fails the read/write with EINVAL. Buffers
// come from Python and carry no alignment guarantee, so fall back to
// buffered I/O whenever either constraint is unmet.
bool odirect_eligible(const void* buf, size_t len, size_t disk_block_size) {
  return disk_block_size > 0 && len % disk_block_size == 0 &&
         reinterpret_cast<uintptr_t>(buf) % disk_block_size == 0;
}
}  // namespace

namespace lmcache {
namespace connector {

uint32_t crc32c_update(uint32_t state, const void* data, size_t len) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
#ifdef LMCACHE_HAVE_SSE42_CRC
  if (crc32c_hardware_available()) {
    return crc32c_hardware(state, p, len);
  }
#endif
  return crc32c_software(state, p, len);
}

uint32_t crc32c(const void* data, size_t len) {
  return crc32c_finish(crc32c_update(crc32c_begin(), data, len));
}

// ---------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------

std::string FSConnector::replace_all(const std::string& str,
                                     const std::string& from,
                                     const std::string& to) {
  std::string result = str;
  size_t pos = 0;
  while ((pos = result.find(from, pos)) != std::string::npos) {
    result.replace(pos, from.size(), to);
    pos += to.size();
  }
  return result;
}

std::string FSConnector::key_to_filename(const std::string& key) {
  // Input key format (from _object_key_to_string):
  //   Legacy unsalted: <model_name>@<kv_rank_hex>@<chunk_hash_hex>
  //   Legacy salted  : <model_name>@<kv_rank_hex>@<chunk_hash_hex>@<cache_salt>
  //   Current unsalted:
  //     <model_name>@<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>
  //   Current salted appends @<cache_salt> to the current unsalted shape.
  //
  // Output filename (matching fs_l2_adapter.py._object_key_to_filename):
  //   Unsalted: <model_name_safe>@0x<kv_rank_hex>@<chunk_hash_hex>.data
  //   Salted  :
  //   <model_name_safe>@0x<kv_rank_hex>@<chunk_hash_hex>@<cache_salt>.data
  //
  // Four fields are intentionally not interpreted: that shape can mean a
  // legacy salted key or a current unsalted key, and both map correctly by
  // preserving every field after kv_rank verbatim.

  // Split on '@'. Current ObjectKey serialization has four or five fields;
  // three-field legacy keys remain readable for cache compatibility.
  std::vector<std::string> parts;
  size_t start = 0;
  for (size_t pos = 0; pos <= key.size(); ++pos) {
    if (pos == key.size() || key[pos] == KEY_SEP) {
      parts.emplace_back(key.substr(start, pos - start));
      start = pos + 1;
    }
  }
  if (parts.size() < 3 || parts.size() > 5) {
    throw std::runtime_error(
        "FSConnector: malformed key (expected 3 to 5 '@'-separated fields): " +
        key);
  }

  const std::string& model_name = parts[0];
  const std::string& kv_rank_hex = parts[1];

  // Replace '/' with '-SEP-' for filesystem safety
  std::string safe_model = replace_all(model_name, "/", PATH_SLASH_REPLACEMENT);

  // Emit the model and normalized rank, preserving all remaining fields.
  std::string result;
  result.reserve(key.size() + 16);
  result += safe_model;
  result += KEY_SEP;
  result += "0x";
  result += kv_rank_hex;
  for (size_t i = 2; i < parts.size(); ++i) {
    result += KEY_SEP;
    result += parts[i];
  }
  result += FILE_EXT;
  return result;
}

// ---------------------------------------------------------------
// read/write helpers
// ---------------------------------------------------------------

static void write_all(int fd, const void* data, size_t len) {
  size_t written = 0;
  const char* ptr = static_cast<const char*>(data);
  while (written < len) {
    ssize_t n = ::write(fd, ptr + written, len - written);
    if (n < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("write failed: " + std::string(strerror(errno)));
    }
    if (n == 0) {
      throw std::runtime_error("write returned 0");
    }
    written += static_cast<size_t>(n);
  }
}

static size_t read_all(int fd, void* buf, size_t len) {
  size_t total = 0;
  char* ptr = static_cast<char*>(buf);
  while (total < len) {
    ssize_t n = ::read(fd, ptr + total, len - total);
    if (n < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("read failed: " + std::string(strerror(errno)));
    }
    if (n == 0) break;  // EOF
    total += static_cast<size_t>(n);
  }
  return total;
}

// ---------------------------------------------------------------
// FSConnector
// ---------------------------------------------------------------

FSConnector::FSConnector(std::string base_path, int num_workers,
                         std::string relative_tmp_dir, bool use_odirect,
                         size_t read_ahead_size)
    : ConnectorBase(num_workers),
      base_path_(std::move(base_path)),
      relative_tmp_dir_(std::move(relative_tmp_dir)),
      use_odirect_(use_odirect),
      disk_block_size_(0),
      read_ahead_size_(read_ahead_size) {
  // Create base directory
  std::filesystem::create_directories(base_path_);

  // Create tmp directory if configured
  if (!relative_tmp_dir_.empty()) {
    auto tmp_path = std::filesystem::path(base_path_) / relative_tmp_dir_;
    std::filesystem::create_directories(tmp_path);
  }

  // Query disk block size for O_DIRECT
  if (use_odirect_) {
    struct statvfs st;
    if (statvfs(base_path_.c_str(), &st) == 0) {
      disk_block_size_ = st.f_bsize;
    }
  }

  start_workers();  // IMPORTANT: call at END of constructor
}

FSConnector::~FSConnector() { close(); }

WorkerFSConn FSConnector::create_connection() {
  WorkerFSConn conn;
  conn.base_path = base_path_;
  if (!relative_tmp_dir_.empty()) {
    conn.tmp_dir = std::filesystem::path(base_path_) / relative_tmp_dir_;
  }
  conn.use_odirect = use_odirect_;
  conn.disk_block_size = disk_block_size_;
  conn.read_ahead_size = read_ahead_size_;
  return conn;
}

void FSConnector::do_single_get(WorkerFSConn& conn, const std::string& key,
                                void* buf, size_t len, size_t chunk_size) {
  std::string filename = key_to_filename(key);
  auto file_path = conn.base_path / filename;

  int flags = O_RDONLY;
  bool do_odirect = conn.use_odirect;
  if (do_odirect) {
    const bool split_aligned = conn.disk_block_size > 0 &&
                               (conn.read_ahead_size == 0 ||
                                len <= conn.read_ahead_size ||
                                conn.read_ahead_size % conn.disk_block_size == 0);
    if (odirect_eligible(buf, len, conn.disk_block_size) && split_aligned) {
#ifdef O_DIRECT
      flags |= O_DIRECT;
#endif
    } else {
      do_odirect = false;
    }
  }

  int fd = ::open(file_path.c_str(), flags);
  if (fd < 0) {
    throw std::runtime_error("open for read failed: " + file_path.string() +
                             ": " + strerror(errno));
  }

  // The payload is read in slices and each slice is folded into the
  // checksum while it is still cache-resident, so verification costs one
  // pass over cache-hot memory instead of a second pass over the whole
  // object. A configured read_ahead_size sets the first slice so the
  // filesystem readahead is triggered as before.
  uint32_t crc_state = crc32c_begin();
  try {
    size_t n = 0;
    char* out = static_cast<char*>(buf);
    size_t first = (conn.read_ahead_size > 0 && len > conn.read_ahead_size)
                       ? conn.read_ahead_size
                       : std::min(len, READ_SLICE_BYTES);
    if (do_odirect && conn.disk_block_size > 0) {
      first = std::max(first, conn.disk_block_size) / conn.disk_block_size *
              conn.disk_block_size;
    }
    while (n < len) {
      size_t want = (n == 0) ? first : std::min(len - n, READ_SLICE_BYTES);
      if (do_odirect && conn.disk_block_size > 0 && n + want < len) {
        want = want / conn.disk_block_size * conn.disk_block_size;
      }
      size_t got = read_all(fd, out + n, want);
      crc_state = crc32c_update(crc_state, out + n, got);
      n += got;
      if (got < want) break;  // EOF
    }
    if (n != len) {
      throw std::runtime_error("incomplete read for " + file_path.string() +
                               ": expected " + std::to_string(len) + ", got " +
                               std::to_string(n));
    }
  } catch (...) {
    ::close(fd);
    throw;
  }
  ::close(fd);
  verify_trailer(file_path, len, crc32c_finish(crc_state));
}

// Verify the integrity trailer of a published object against the checksum
// of the payload that was just read. The trailer is read through a separate
// buffered descriptor so O_DIRECT alignment rules never apply to it.
void FSConnector::verify_trailer(const std::filesystem::path& file_path,
                                 size_t len, uint32_t payload_crc) {
  struct stat st;
  if (::stat(file_path.c_str(), &st) != 0) {
    throw std::runtime_error("stat failed: " + file_path.string() + ": " +
                             strerror(errno));
  }
  const uint64_t file_size = static_cast<uint64_t>(st.st_size);
  if (file_size == len) {
    return;  // object written before trailers existed: accepted unverified
  }
  if (file_size != len + sizeof(ObjectTrailer)) {
    throw std::runtime_error("object size mismatch for " + file_path.string() +
                             ": payload " + std::to_string(len) + ", file " +
                             std::to_string(file_size));
  }
  int fd = ::open(file_path.c_str(), O_RDONLY);
  if (fd < 0) {
    throw std::runtime_error("open for trailer failed: " + file_path.string() +
                             ": " + strerror(errno));
  }
  ObjectTrailer trailer;
  ssize_t got = ::pread(fd, &trailer, sizeof(trailer), static_cast<off_t>(len));
  ::close(fd);
  if (got != static_cast<ssize_t>(sizeof(trailer))) {
    throw std::runtime_error("trailer read failed for " + file_path.string());
  }
  if (trailer.magic != OBJECT_TRAILER_MAGIC || trailer.payload_len != len) {
    throw std::runtime_error("trailer mismatch for " + file_path.string());
  }
  if (payload_crc != trailer.crc32c) {
    throw std::runtime_error("checksum mismatch for " + file_path.string() +
                             ": stored " + std::to_string(trailer.crc32c) +
                             ", computed " + std::to_string(payload_crc));
  }
}

void FSConnector::do_single_set(WorkerFSConn& conn, const std::string& key,
                                const void* buf, size_t len,
                                size_t chunk_size) {
  std::string filename = key_to_filename(key);
  auto file_path = conn.base_path / filename;

  // Skip if already stored on disk
  if (std::filesystem::exists(file_path)) {
    return;
  }

  const auto tmp_dir =
      conn.tmp_dir.empty() ? file_path.parent_path() : conn.tmp_dir;
  std::filesystem::path tmp_path;
  int flags = O_CREAT | O_EXCL | O_WRONLY;
  bool do_odirect = conn.use_odirect;
  if (do_odirect) {
    if (odirect_eligible(buf, len, conn.disk_block_size)) {
#ifdef O_DIRECT
      flags |= O_DIRECT;
#endif
    } else {
      do_odirect = false;
    }
  }

  int fd = -1;
  for (size_t attempt = 0; attempt < 1024; ++attempt) {
    const uint64_t id =
        next_temp_file_id.fetch_add(1, std::memory_order_relaxed);
    tmp_path = tmp_dir / (filename + TMP_EXT + "." +
                          std::to_string(static_cast<uint64_t>(::getpid())) +
                          "." + std::to_string(id));
    fd = ::open(tmp_path.c_str(), flags, 0644);
    if (fd >= 0) break;
    if (errno != EEXIST) {
      throw std::runtime_error("open for write failed: " + tmp_path.string() +
                               ": " + strerror(errno));
    }
  }
  if (fd < 0) {
    throw std::runtime_error("failed to allocate a unique temporary file for " +
                             file_path.string());
  }

  uint32_t crc_state = crc32c_begin();
  try {
    // Written in slices; each slice is folded into the checksum right after
    // it is handed to the kernel, while it is still cache-resident.
    const char* in = static_cast<const char*>(buf);
    size_t off = 0;
    while (off < len) {
      size_t want = std::min(len - off, READ_SLICE_BYTES);
      if (do_odirect && conn.disk_block_size > 0 && off + want < len) {
        want = want / conn.disk_block_size * conn.disk_block_size;
      }
      write_all(fd, in + off, want);
      crc_state = crc32c_update(crc_state, in + off, want);
      off += want;
    }
    if (::close(fd) != 0) {
      const int close_errno = errno;
      fd = -1;
      throw std::runtime_error("close after write failed: " +
                               std::string(strerror(close_errno)));
    }
    fd = -1;
    append_trailer_and_sync(tmp_path, len, crc32c_finish(crc_state));
  } catch (...) {
    if (fd >= 0) ::close(fd);
    std::filesystem::remove(tmp_path);
    throw;
  }

  // A hard link publishes the completed inode without replacing an existing
  // value. Concurrent or cross-process writers therefore never share writable
  // storage and a reader can only observe a complete file.
  if (::link(tmp_path.c_str(), file_path.c_str()) != 0) {
    const int link_errno = errno;
    std::error_code remove_ec;
    std::filesystem::remove(tmp_path, remove_ec);
    if (link_errno == EEXIST) return;
    throw std::runtime_error("publish failed: " + tmp_path.string() + " -> " +
                             file_path.string() + ": " + strerror(link_errno));
  }

  std::error_code remove_ec;
  std::filesystem::remove(tmp_path, remove_ec);
  if (remove_ec) {
    fprintf(stderr, "[LMCache SET] temporary file cleanup failed: %s: %s\n",
            tmp_path.c_str(), remove_ec.message().c_str());
  }
  sync_directory(file_path.parent_path());
}

// Append the integrity trailer through a buffered descriptor (the payload
// may have been written with O_DIRECT, whose alignment rules the 16-byte
// trailer cannot meet) and make the whole inode durable before it is
// published. A crash between the payload write and the publish leaves only
// an unlinked temporary file behind.
void FSConnector::append_trailer_and_sync(const std::filesystem::path& tmp_path,
                                          size_t len, uint32_t payload_crc) {
  ObjectTrailer trailer;
  trailer.magic = OBJECT_TRAILER_MAGIC;
  trailer.crc32c = payload_crc;
  trailer.payload_len = static_cast<uint64_t>(len);
  int fd = ::open(tmp_path.c_str(), O_WRONLY | O_APPEND);
  if (fd < 0) {
    throw std::runtime_error("open for trailer failed: " + tmp_path.string() +
                             ": " + strerror(errno));
  }
  try {
    write_all(fd, &trailer, sizeof(trailer));
    if (::fsync(fd) != 0) {
      throw std::runtime_error("fsync failed: " + tmp_path.string() + ": " +
                               strerror(errno));
    }
  } catch (...) {
    ::close(fd);
    throw;
  }
  if (::close(fd) != 0) {
    throw std::runtime_error("close after trailer failed: " +
                             tmp_path.string() + ": " + strerror(errno));
  }
}

// Make a published directory entry durable. Failure here does not affect
// the object's correctness (the file is complete and verified on load), so
// it is reported rather than raised.
void FSConnector::sync_directory(const std::filesystem::path& dir) {
  int dfd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY);
  if (dfd < 0) {
    fprintf(stderr, "[LMCache SET] directory open for fsync failed: %s: %s\n",
            dir.c_str(), strerror(errno));
    return;
  }
  if (::fsync(dfd) != 0) {
    fprintf(stderr, "[LMCache SET] directory fsync failed: %s: %s\n",
            dir.c_str(), strerror(errno));
  }
  ::close(dfd);
}

bool FSConnector::do_single_exists(WorkerFSConn& conn, const std::string& key) {
  std::string filename = key_to_filename(key);
  auto file_path = conn.base_path / filename;
  return std::filesystem::exists(file_path);
}

bool FSConnector::do_single_delete(WorkerFSConn& conn, const std::string& key) {
  std::string filename = key_to_filename(key);
  auto file_path = conn.base_path / filename;
  std::error_code ec;
  return std::filesystem::remove(file_path, ec);
}

}  // namespace connector
}  // namespace lmcache
