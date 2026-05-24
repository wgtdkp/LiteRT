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

#include "litert/tools/flags/vendors/hailo_flags.h"

#include <cstdlib>
#include <string>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "litert/cc/litert_expected.h"

ABSL_FLAG(std::string, hailo_hef_path, "",
          "Path to a pre-compiled Hailo HEF file. When set, the Hailo "
          "compiler plugin will embed this HEF into the output .tflite "
          "model so that inference is dispatched to the Hailo NPU via "
          "the LiteRT dispatch API.");

namespace litert::hailo {

Expected<void> UpdateHailoFromFlags() {
  const std::string hef_path = absl::GetFlag(FLAGS_hailo_hef_path);
  if (!hef_path.empty()) {
    ::setenv("LITERT_HAILO_HEF_PATH", hef_path.c_str(), /*overwrite=*/1);
  }
  return {};
}

}  // namespace litert::hailo
