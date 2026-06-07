#ifndef ODML_LITERT_LITERT_VENDORS_HAILO_DISPATCH_INVOCATION_CONTEXT_H_
#define ODML_LITERT_LITERT_VENDORS_HAILO_DISPATCH_INVOCATION_CONTEXT_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <utility>

#include "litert/c/litert_common.h"
#include "litert/c/internal/litert_runtime_context.h"
#include "litert/cc/litert_expected.h"
#include "litert/vendors/c/litert_dispatch.h"
#include "litert/vendors/hailo/dispatch/device_context.h"

#include "hailo/hailort.hpp"

class LiteRtDispatchInvocationContextT {
 public:
  using Ptr = std::unique_ptr<LiteRtDispatchInvocationContextT>;

  ~LiteRtDispatchInvocationContextT() = default;

  static litert::Expected<Ptr> Create(
      LiteRtDispatchDeviceContextT& device_context,
      LiteRtDispatchExecutableType exec_type,
      const LiteRtMemBuffer* exec_bytecode_buffer,
      const char* function_name,
      int num_inputs,
      int num_outputs);

  litert::Expected<void> AttachInput(int index, LiteRtTensorBufferHandle handle);
  litert::Expected<void> AttachOutput(int index, LiteRtTensorBufferHandle handle);
  litert::Expected<void> DetachInput(int index, LiteRtTensorBufferHandle handle);
  litert::Expected<void> DetachOutput(int index, LiteRtTensorBufferHandle handle);

  litert::Expected<void> Invoke();

  // Requirements negotiation
  litert::Expected<LiteRtTensorBufferRequirements> GetInputRequirements(
      int input_index, const LiteRtRankedTensorType& tensor_type);
  litert::Expected<LiteRtTensorBufferRequirements> GetOutputRequirements(
      int output_index, const LiteRtRankedTensorType& tensor_type);

 private:
  explicit LiteRtDispatchInvocationContextT(
      LiteRtDispatchDeviceContextT& device_context,
            std::shared_ptr<hailort::InferModel> infer_model,
            hailort::ConfiguredInferModel configured_infer_model,
            std::vector<std::string> input_names,
            std::vector<std::string> output_names)
      : device_context_(device_context),
                infer_model_(std::move(infer_model)),
                configured_infer_model_(std::move(configured_infer_model)),
                input_names_(std::move(input_names)),
                output_names_(std::move(output_names)) {
        attached_inputs_.resize(input_names_.size(), (LiteRtTensorBufferHandle)0);
        attached_outputs_.resize(output_names_.size(), (LiteRtTensorBufferHandle)0);
  }

  LiteRtDispatchDeviceContextT& device_context_;
    std::shared_ptr<hailort::InferModel> infer_model_;
    hailort::ConfiguredInferModel configured_infer_model_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;

  std::vector<LiteRtTensorBufferHandle> attached_inputs_;
  std::vector<LiteRtTensorBufferHandle> attached_outputs_;
};

#endif  // ODML_LITERT_LITERT_VENDORS_HAILO_DISPATCH_INVOCATION_CONTEXT_H_
