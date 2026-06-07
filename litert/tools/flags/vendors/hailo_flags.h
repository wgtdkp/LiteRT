// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_VENDORS_HAILO_FLAGS_H_
#define THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_VENDORS_HAILO_FLAGS_H_

#include <string>

#include "absl/flags/declare.h"  // from @com_google_absl
#include "litert/cc/litert_expected.h"

// COMPILATION OPTIONS /////////////////////////////////////////////////////////

// Path to a pre-compiled Hailo HEF file. The Hailo compiler plugin reads
// this via the LITERT_HAILO_HEF_PATH environment variable to embed the HEF
// bytecode into the output .tflite model, enabling the apply/compile commands
// to produce a LiteRT-compatible flatbuffer that dispatches inference on Hailo.
ABSL_DECLARE_FLAG(std::string, hailo_hef_path);

// PARSERS (internal) //////////////////////////////////////////////////////////

namespace litert::hailo {

// Propagates Hailo-specific command-line flags to the runtime environment.
// Sets LITERT_HAILO_HEF_PATH from --hailo_hef_path so the Hailo compiler
// plugin can locate the pre-compiled HEF file.
Expected<void> UpdateHailoFromFlags();

}  // namespace litert::hailo

#endif  // THIRD_PARTY_ODML_LITERT_LITERT_TOOLS_FLAGS_VENDORS_HAILO_FLAGS_H_
