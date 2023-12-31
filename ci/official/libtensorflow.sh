#!/bin/bash
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
source "${BASH_SOURCE%/*}/utilities/setup.sh"

# Record GPU count and CUDA version status
if [[ "$TFCI_NVIDIA_SMI_ENABLE" == 1 ]]; then
  tfrun nvidia-smi
fi

# Update the version numbers for Nightly only
if [[ "$TFCI_NIGHTLY_UPDATE_VERSION_ENABLE" == 1 ]]; then
  tfrun python3 tensorflow/tools/ci_build/update_version.py --nightly
fi

tfrun bazel "${TFCI_BAZEL_BAZELRC_ARGS[@]}" test "${TFCI_BAZEL_COMMON_ARGS[@]}" --config=libtensorflow_test
tfrun bazel "${TFCI_BAZEL_BAZELRC_ARGS[@]}" build "${TFCI_BAZEL_COMMON_ARGS[@]}" --config=libtensorflow_build

tfrun ./ci/official/utilities/repack_libtensorflow.sh build "$TFCI_LIB_SUFFIX"

if [[ "$TFCI_UPLOAD_LIB_ENABLE" == 1 ]]; then
  gsutil cp build/*.tar.gz "$TFCI_UPLOAD_LIB_GCS_URI"
  if [[ "$TFCI_UPLOAD_LIB_LATEST_ENABLE" == 1 ]]; then
    gsutil cp build/*.tar.gz "$TFCI_UPLOAD_LIB_LATEST_GCS_URI"
  fi
fi
