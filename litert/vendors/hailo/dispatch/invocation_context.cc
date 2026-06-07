#include "litert/vendors/hailo/dispatch/invocation_context.h"

#include <vector>
#include <string>
#include <memory>
#include <cstring>
#include <chrono>

#include "litert/c/internal/litert_logging.h"
#include "litert/cc/litert_expected.h"
#include "litert/cc/litert_macros.h"
#include "litert/core/util/tensor_type_util.h"

litert::Expected<LiteRtDispatchInvocationContextT::Ptr>
LiteRtDispatchInvocationContextT::Create(
    LiteRtDispatchDeviceContextT& device_context,
    LiteRtDispatchExecutableType exec_type,
    const LiteRtMemBuffer* exec_bytecode_buffer,
    const char* function_name,
    int num_inputs,
    int num_outputs) {
  if (exec_bytecode_buffer == nullptr || exec_bytecode_buffer->base_addr == nullptr) {
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument, "Invalid exec bytecode buffer");
  }

  // Load HEF from buffer memory view.
  const void* hef_ptr = static_cast<const uint8_t*>(exec_bytecode_buffer->base_addr) +
                        exec_bytecode_buffer->offset;
  auto hef_exp = hailort::Hef::create(
      hailort::MemoryView(const_cast<void*>(hef_ptr), exec_bytecode_buffer->size));
  if (!hef_exp) {
    LITERT_LOG(LITERT_ERROR, "Failed to load HEF from buffer: status = %d", hef_exp.status());
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to load HEF");
  }
  auto infer_model_exp = device_context.vdevice().create_infer_model(
      hailort::MemoryView(const_cast<void*>(hef_ptr), exec_bytecode_buffer->size));
  if (!infer_model_exp) {
    LITERT_LOG(LITERT_ERROR, "Failed to create infer model: status = %d", infer_model_exp.status());
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to create infer model");
  }

  auto infer_model = std::move(infer_model_exp.value());
  for (const auto& input_name : infer_model->get_input_names()) {
    auto infer_input_exp = infer_model->input(input_name);
    if (!infer_input_exp) {
      LITERT_LOG(LITERT_ERROR, "Failed to get infer model input '%s': status = %d",
                 input_name.c_str(), infer_input_exp.status());
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                "Failed to get infer model input stream");
    }
    auto infer_input = std::move(infer_input_exp.value());
    infer_input.set_format_type(HAILO_FORMAT_TYPE_FLOAT32);
  }

  for (const auto& output_name : infer_model->get_output_names()) {
    auto infer_output_exp = infer_model->output(output_name);
    if (!infer_output_exp) {
      LITERT_LOG(LITERT_ERROR, "Failed to get infer model output '%s': status = %d",
                 output_name.c_str(), infer_output_exp.status());
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure,
                                "Failed to get infer model output stream");
    }
    auto infer_output = std::move(infer_output_exp.value());
    infer_output.set_format_type(HAILO_FORMAT_TYPE_FLOAT32);
  }

  auto configured_infer_model_exp = infer_model->configure();
  if (!configured_infer_model_exp) {
    LITERT_LOG(LITERT_ERROR, "Failed to configure infer model: status = %d", configured_infer_model_exp.status());
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to configure infer model");
  }

  auto configured_infer_model = std::move(configured_infer_model_exp.value());
  auto input_names = infer_model->get_input_names();
  auto output_names = infer_model->get_output_names();

  if (static_cast<int>(input_names.size()) != num_inputs) {
    LITERT_LOG(LITERT_ERROR, "Mismatch in inputs: model has %d, LiteRT has %d",
               static_cast<int>(input_names.size()), num_inputs);
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument, "Inputs count mismatch");
  }

  if (static_cast<int>(output_names.size()) != num_outputs) {
    LITERT_LOG(LITERT_ERROR, "Mismatch in outputs: model has %d, LiteRT has %d",
               static_cast<int>(output_names.size()), num_outputs);
    return litert::Unexpected(kLiteRtStatusErrorInvalidArgument, "Outputs count mismatch");
  }

  LITERT_LOG(LITERT_INFO, "Hailo InvocationContext initialized successfully.");

  return Ptr(new LiteRtDispatchInvocationContextT(
      device_context, std::move(infer_model), std::move(configured_infer_model),
      std::move(input_names), std::move(output_names)));
}

litert::Expected<void>
LiteRtDispatchInvocationContextT::AttachInput(int index, LiteRtTensorBufferHandle handle) {
  if (index < 0 || index >= static_cast<int>(attached_inputs_.size())) {
    return litert::Unexpected(kLiteRtStatusErrorIndexOOB, "Input index out of bounds");
  }
  attached_inputs_[index] = handle;
  return {};
}

litert::Expected<void>
LiteRtDispatchInvocationContextT::AttachOutput(int index, LiteRtTensorBufferHandle handle) {
  if (index < 0 || index >= static_cast<int>(attached_outputs_.size())) {
    return litert::Unexpected(kLiteRtStatusErrorIndexOOB, "Output index out of bounds");
  }
  attached_outputs_[index] = handle;
  return {};
}

litert::Expected<void>
LiteRtDispatchInvocationContextT::DetachInput(int index, LiteRtTensorBufferHandle handle) {
  if (index < 0 || index >= static_cast<int>(attached_inputs_.size())) {
    return litert::Unexpected(kLiteRtStatusErrorIndexOOB, "Input index out of bounds");
  }
  attached_inputs_[index] = (LiteRtTensorBufferHandle)0;
  return {};
}

litert::Expected<void>
LiteRtDispatchInvocationContextT::DetachOutput(int index, LiteRtTensorBufferHandle handle) {
  if (index < 0 || index >= static_cast<int>(attached_outputs_.size())) {
    return litert::Unexpected(kLiteRtStatusErrorIndexOOB, "Output index out of bounds");
  }
  attached_outputs_[index] = (LiteRtTensorBufferHandle)0;
  return {};
}

litert::Expected<void>
LiteRtDispatchInvocationContextT::Invoke() {
  // Validate attachments.
  for (size_t i = 0; i < attached_inputs_.size(); ++i) {
    if (attached_inputs_[i] == (LiteRtTensorBufferHandle)0) {
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, "Input stream not attached");
    }
  }
  for (size_t i = 0; i < attached_outputs_.size(); ++i) {
    if (attached_outputs_[i] == (LiteRtTensorBufferHandle)0) {
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, "Output stream not attached");
    }
  }

  auto bindings_exp = configured_infer_model_.create_bindings();
  if (!bindings_exp) {
    LITERT_LOG(LITERT_ERROR, "Failed to create infer bindings: status = %d", bindings_exp.status());
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to create infer bindings");
  }

  auto bindings = std::move(bindings_exp.value());

  // Bind LiteRT input buffers to Hailo infer model inputs.
  for (size_t i = 0; i < input_names_.size(); ++i) {
    LITERT_ASSIGN_OR_RETURN(void* host_addr, device_context_.GetHostMemoryAddress(attached_inputs_[i]));
    LITERT_ASSIGN_OR_RETURN(const size_t frame_size, device_context_.GetHostMemorySize(attached_inputs_[i]));
    auto input_stream_exp = bindings.input(input_names_[i]);
    if (!input_stream_exp) {
      LITERT_LOG(LITERT_ERROR, "Failed to get input stream '%s': status = %d",
                 input_names_[i].c_str(), input_stream_exp.status());
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to get input stream");
    }
    auto infer_input_stream = std::move(input_stream_exp.value());
    auto status = infer_input_stream.set_buffer(hailort::MemoryView(host_addr, frame_size));
    if (status != HAILO_SUCCESS) {
      LITERT_LOG(LITERT_ERROR, "Failed to set input buffer '%s': status = %d",
                 input_names_[i].c_str(), status);
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to set input buffer");
    }
  }

  // Bind LiteRT output buffers to Hailo infer model outputs.
  for (size_t i = 0; i < output_names_.size(); ++i) {
    LITERT_ASSIGN_OR_RETURN(void* host_addr, device_context_.GetHostMemoryAddress(attached_outputs_[i]));
    LITERT_ASSIGN_OR_RETURN(const size_t frame_size, device_context_.GetHostMemorySize(attached_outputs_[i]));
    auto output_stream_exp = bindings.output(output_names_[i]);
    if (!output_stream_exp) {
      LITERT_LOG(LITERT_ERROR, "Failed to get output stream '%s': status = %d",
                 output_names_[i].c_str(), output_stream_exp.status());
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to get output stream");
    }
    auto infer_output_stream = std::move(output_stream_exp.value());
    auto status = infer_output_stream.set_buffer(hailort::MemoryView(host_addr, frame_size));
    if (status != HAILO_SUCCESS) {
      LITERT_LOG(LITERT_ERROR, "Failed to set output buffer '%s': status = %d",
                 output_names_[i].c_str(), status);
      return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to set output buffer");
    }
  }

  const auto run_status = configured_infer_model_.run(bindings, std::chrono::milliseconds(10000));
  if (run_status != HAILO_SUCCESS) {
    LITERT_LOG(LITERT_ERROR, "Failed to run infer model: status = %d", run_status);
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to run infer model");
  }

  return {};
}

litert::Expected<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::GetInputRequirements(
    int input_index, const LiteRtRankedTensorType& tensor_type) {
  LiteRtTensorBufferType supported_tensor_buffer_types[] = {
      kLiteRtTensorBufferTypeHostMemory,
  };
  int num_supported_types = sizeof(supported_tensor_buffer_types) / sizeof(supported_tensor_buffer_types[0]);

  auto buffer_size = litert::internal::GetNumPackedBytes(tensor_type);
  if (!buffer_size) {
    return litert::Unexpected(buffer_size.Error());
  }

  LiteRtTensorBufferRequirements requirements;
  auto status = device_context_.runtime_context()->create_tensor_buffer_requirements(
      num_supported_types, supported_tensor_buffer_types, *buffer_size, 0, nullptr, &requirements);
  if (status != kLiteRtStatusOk) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to create tensor buffer requirements");
  }

  return requirements;
}

litert::Expected<LiteRtTensorBufferRequirements>
LiteRtDispatchInvocationContextT::GetOutputRequirements(
    int output_index, const LiteRtRankedTensorType& tensor_type) {
  LiteRtTensorBufferType supported_tensor_buffer_types[] = {
      kLiteRtTensorBufferTypeHostMemory,
  };
  int num_supported_types = sizeof(supported_tensor_buffer_types) / sizeof(supported_tensor_buffer_types[0]);

  auto buffer_size = litert::internal::GetNumPackedBytes(tensor_type);
  if (!buffer_size) {
    return litert::Unexpected(buffer_size.Error());
  }

  LiteRtTensorBufferRequirements requirements;
  auto status = device_context_.runtime_context()->create_tensor_buffer_requirements(
      num_supported_types, supported_tensor_buffer_types, *buffer_size, 0, nullptr, &requirements);
  if (status != kLiteRtStatusOk) {
    return litert::Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to create tensor buffer requirements");
  }

  return requirements;
}
