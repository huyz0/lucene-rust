// A C ABI over libtorch's TorchScript runtime, for lucene-ffi's ml_torch
// module: load a module, run it on token arrays the way DJL's PyTorch engine
// feeds them, and read every output tensor back.
//
// Inputs follow DJL's IValueUtils: an array named "<group>.<key>" goes into a
// string-keyed dict passed as one argument (at the position of the group's
// first array); any other array is a positional tensor argument. Each array
// is a [1, len] int64 tensor (a batch of one).
//
// Outputs are flattened the way DJL's IValue.toNDList does: a tensor, a list
// or tuple (recursively) and a one-level dict of tensors, whose keys become
// the output names. Each dict's entries carry the dict's number (from 1), so
// the caller can order them as DJL's map does.
//
// Built by scripts/build-torch-shim.sh, against libtorch's headers, once per
// C++ string ABI (libtorch ships both); it links to no library at build
// time: the caller loads libtorch with global symbols first.

#include <ATen/Parallel.h>
#include <torch/script.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

// libtorch's CPU builds parallelize with OpenMP (libgomp, loaded with them).
extern "C" void omp_set_num_threads(int);

namespace {

thread_local std::string last_error;

// The intra-op threads one forward call may use when it runs alone, and the
// calls running now. Each call's OpenMP team is its calling thread's own, so
// concurrent calls split the threads rather than each taking all of them.
int max_threads = 1;
std::atomic<int> running{0};

struct Running {
  int now;
  Running() : now(running.fetch_add(1) + 1) {}
  ~Running() { running.fetch_sub(1); }
};

struct Output {
  std::string name;
  uint32_t group;
  at::Tensor tensor;
  std::vector<int64_t> shape;
};

struct Result {
  std::vector<Output> outputs;
};

void flatten(const c10::IValue& v, Result& r, uint32_t& groups) {
  if (v.isTensor()) {
    r.outputs.push_back({"", 0, v.toTensor(), {}});
  } else if (v.isTensorList()) {
    for (const at::Tensor& t : v.toTensorVector()) r.outputs.push_back({"", 0, t, {}});
  } else if (v.isGenericDict()) {
    uint32_t group = ++groups;
    for (const auto& e : v.toGenericDict()) {
      if (!e.value().isTensor()) throw std::runtime_error("Only one level of map is supported.");
      r.outputs.push_back({e.key().toStringRef(), group, e.value().toTensor(), {}});
    }
  } else if (v.isList()) {
    for (const c10::IValue& e : v.toListRef()) flatten(e, r, groups);
  } else if (v.isTuple()) {
    for (const c10::IValue& e : v.toTupleRef().elements()) flatten(e, r, groups);
  } else {
    throw std::runtime_error("Unsupported IValue type.");
  }
}

}  // namespace

#define EXPORT __attribute__((visibility("default")))

// Newer compilers' headers reference these from libstdc++ and glibc versions
// older systems lack (GCC 11 and 13, glibc 2.32); local definitions keep the
// library loadable wherever libtorch itself loads. The stream library
// initializes its own streams, and a process taken for multi-threaded only
// makes shared pointers count atomically.
namespace std {
[[noreturn]] void __throw_bad_array_new_length() {
  throw std::bad_array_new_length();
}
void ios_base_library_init() {}
}  // namespace std
extern "C" {
char __libc_single_threaded = 0;
}

extern "C" {

// The data types of an output: 0 float32, 1 int64.
struct lucene_torch_output {
  const char* name;
  size_t name_len;
  uint32_t group;
  int32_t dtype;
  int32_t rank;
  const int64_t* shape;
  const void* data;
  int64_t numel;
};

EXPORT const char* lucene_torch_last_error() { return last_error.c_str(); }

// The intra-op threads a forward call running alone uses (0: libtorch's
// default, one per physical core), shared out among calls that run at once,
// and one inter-op thread, as DJL sets it. Returns 0, or -1 with the error in
// lucene_torch_last_error.
EXPORT int32_t lucene_torch_set_threads(int32_t intra) {
  try {
    if (intra > 0) at::set_num_threads(intra);
    max_threads = std::max(1, at::get_num_threads());
    if (at::get_num_interop_threads() != 1) at::set_num_interop_threads(1);
    return 0;
  } catch (const std::exception& e) {
    last_error = e.what();
    return -1;
  }
}

EXPORT void* lucene_torch_load(const char* path, size_t path_len) {
  try {
    auto m = std::make_unique<torch::jit::Module>(torch::jit::load(std::string(path, path_len)));
    m->eval();
    return m.release();
  } catch (const std::exception& e) {
    last_error = e.what();
    return nullptr;
  }
}

EXPORT void lucene_torch_free(void* module) { delete static_cast<torch::jit::Module*>(module); }

// Runs `module` on `n` arrays of `len` values each, named by names[i]
// (name_lens[i] bytes). Returns a result to read with lucene_torch_output and
// free with lucene_torch_result_free, or null with the error set.
EXPORT void* lucene_torch_forward(void* module, int32_t n, const char* const* names, const size_t* name_lens,
                           const int64_t* const* arrays, int64_t len) {
  try {
    thread_local bool threads_initialized = false;
    if (!threads_initialized) {
      at::init_num_threads();
      threads_initialized = true;
    }
    Running call;
    omp_set_num_threads(std::max(1, max_threads / call.now));
    c10::InferenceMode guard;
    std::vector<c10::IValue> args;
    std::vector<std::string> group_names;
    std::vector<size_t> group_slots;
    std::vector<c10::Dict<std::string, at::Tensor>> dicts;
    for (int32_t i = 0; i < n; i++) {
      at::Tensor t = torch::from_blob(const_cast<int64_t*>(arrays[i]), {1, len}, torch::kLong).clone();
      std::string name(names[i], name_lens[i]);
      size_t dot = name.find('.');
      if (dot == std::string::npos) {
        args.emplace_back(t);
        continue;
      }
      std::string group = name.substr(0, dot);
      size_t g = 0;
      while (g < group_names.size() && group_names[g] != group) g++;
      if (g == group_names.size()) {
        group_names.push_back(group);
        group_slots.push_back(args.size());
        dicts.emplace_back();
        args.emplace_back();
      }
      dicts[g].insert(name.substr(dot + 1), t);
    }
    for (size_t g = 0; g < dicts.size(); g++) args[group_slots[g]] = c10::IValue(dicts[g]);
    c10::IValue value = static_cast<torch::jit::Module*>(module)->forward(args);
    auto r = std::make_unique<Result>();
    uint32_t groups = 0;
    flatten(value, *r, groups);
    for (Output& o : r->outputs) {
      at::ScalarType type = o.tensor.scalar_type();
      if (type != at::kFloat && type != at::kLong) {
        throw std::runtime_error(std::string("unsupported output data type ") + c10::toString(type));
      }
      o.tensor = o.tensor.to(at::kCPU).contiguous();
      o.shape = o.tensor.sizes().vec();
    }
    return r.release();
  } catch (const std::exception& e) {
    last_error = e.what();
    return nullptr;
  }
}

EXPORT int32_t lucene_torch_outputs(const void* result) {
  return static_cast<int32_t>(static_cast<const Result*>(result)->outputs.size());
}

// Writes output `i` of `result` to `*out`; its pointers stay valid until the
// result is freed. Returns 0, or -1 when `i` is out of range.
EXPORT int32_t lucene_torch_output(const void* result, int32_t i, lucene_torch_output* out) {
  const Result* r = static_cast<const Result*>(result);
  if (i < 0 || static_cast<size_t>(i) >= r->outputs.size()) return -1;
  const Output& o = r->outputs[static_cast<size_t>(i)];
  out->name = o.name.data();
  out->name_len = o.name.size();
  out->group = o.group;
  out->dtype = o.tensor.scalar_type() == at::kFloat ? 0 : 1;
  out->rank = static_cast<int32_t>(o.shape.size());
  out->shape = o.shape.data();
  out->data = o.tensor.data_ptr();
  out->numel = o.tensor.numel();
  return 0;
}

EXPORT void lucene_torch_result_free(void* result) { delete static_cast<Result*>(result); }

}  // extern "C"
