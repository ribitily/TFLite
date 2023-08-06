# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# limitations under the License..
# ==============================================================================
"""TensorFlow is an open source machine learning framework for everyone.

[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)](https://badge.fury.io/py/tensorflow)
[![PyPI](https://badge.fury.io/py/tensorflow.svg)](https://badge.fury.io/py/tensorflow)

TensorFlow is an open source software library for high performance numerical
computation. Its flexible architecture allows easy deployment of computation
across a variety of platforms (CPUs, GPUs, TPUs), and from desktops to clusters
of servers to mobile and edge devices.

Originally developed by researchers and engineers from the Google Brain team
within Google's AI organization, it comes with strong support for machine
learning and deep learning and the flexible numerical computation core is used
across many other scientific domains. TensorFlow is licensed under [Apache
2.0](https://github.com/tensorflow/tensorflow/blob/master/LICENSE).
"""

import fnmatch
import os
import platform
import re
import sys

from setuptools import Command
from setuptools import find_packages
from setuptools import setup
from setuptools.command.install import install as InstallCommandBase
from setuptools.dist import Distribution


# This version string is semver compatible, but incompatible with pip.
# For pip, we will remove all '-' characters from this string, and use the
# result for pip.
# Also update tensorflow/tensorflow.bzl and
# tensorflow/core/public/version.h
_VERSION = '2.15.0'


# We use the same setup.py for all tensorflow_* packages and for the nightly
# equivalents (tf_nightly_*). The package is controlled from the argument line
# when building the pip package.
project_name = 'tensorflow'
if '--project_name' in sys.argv:
  project_name_idx = sys.argv.index('--project_name')
  project_name = sys.argv[project_name_idx + 1]
  sys.argv.remove('--project_name')
  sys.argv.pop(project_name_idx)


collaborator_build = False
if '--collaborator_build' in sys.argv:
  sys.argv.remove('--collaborator_build')
  collaborator_build = True


# Returns standard if a tensorflow-* package is being built, and nightly if a
# tf_nightly-* package is being built.
def standard_or_nightly(standard, nightly):
  return nightly if 'tf_nightly' in project_name else standard

# All versions of TF need these packages. We indicate the widest possible range
# of package releases possible to be as up-to-date as possible as well as to
# accomodate as many pre-installed packages as possible.
# For packages that don't have yet a stable release, we pin using `~= 0.x` which
# means we accept any `0.y` version (y >= x) but not the first major release. We
# will need additional testing for that.
# NOTE: This assumes that all packages follow SemVer. If a package follows a
# different versioning scheme (e.g., PVP), we use different bound specifier and
# comment the versioning scheme.
REQUIRED_PACKAGES = [
    'absl-py >= 1.0.0',
    'astunparse >= 1.6.0',
    'flatbuffers >= 23.5.26',
    'gast >=0.2.1,!=0.5.0,!=0.5.1,!=0.5.2',
    'google_pasta >= 0.1.1',
    'h5py >= 2.9.0',
    'libclang >= 13.0.0',
    'ml_dtypes >= 0.2.0',
    'numpy >= 1.23.5',
    'opt_einsum >= 2.3.2',
    'packaging',
    # pylint:disable=line-too-long
    (
        'protobuf>=3.20.3,<5.0.0dev,!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5'
    ),
    'setuptools',
    'six >= 1.12.0',
    'termcolor >= 1.1.0',
    'typing_extensions >= 3.6.6',
    'wrapt >= 1.11.0, < 1.15',
    # This looks worse as a wrapped line. pylint:disable=line-too-long
    (
        'tensorflow-io-gcs-filesystem >= 0.23.1;platform_machine!="arm64" or'
        ' platform_system!="Darwin"'
    ),
    # grpcio does not build correctly on big-endian machines due to lack of
    # BoringSSL support.
    # See https://github.com/tensorflow/tensorflow/issues/17882.
    'grpcio >= 1.24.3, < 2.0' if sys.byteorder == 'little' else None,
    # TensorFlow exposes the TF API for certain TF ecosystem packages like
    # keras.  When TF depends on those packages, the package version needs to
    # match the current TF version. For tf_nightly, we install the nightly
    # variant of each package instead, which must be one version ahead of the
    # current release version. These also usually have "alpha" or "dev" in their
    # version name.
    # These are all updated during the TF release process.
    standard_or_nightly(
        'tensorboard >= 2.13, < 2.14', 'tb-nightly ~= 2.14.0.a'
    ),
    standard_or_nightly(
        'tensorflow_estimator >= 2.13.0rc0, < 2.14',
        'tf-estimator-nightly ~= 2.14.0.dev',
    ),
    standard_or_nightly(
        'keras >= 2.13.1rc0, < 2.14', 'keras-nightly ~= 2.14.0.dev'
    ),
]
REQUIRED_PACKAGES = [p for p in REQUIRED_PACKAGES if p is not None]

FAKE_REQUIRED_PACKAGES = [
    # The depedencies here below are not actually used but are needed for
    # package managers like poetry to parse as they are confused by the
    # different architectures having different requirements.
    # The entries here should be a simple duplicate of those in the collaborator
    # build section.
    standard_or_nightly('tensorflow-cpu-aws', 'tf-nightly-cpu-aws') + '==' +
    _VERSION + ';platform_system=="Linux" and (platform_machine=="arm64" or '
    'platform_machine=="aarch64")',
    standard_or_nightly('tensorflow-intel', 'tf-nightly-intel') + '==' +
    _VERSION + ';platform_system=="Windows"',
]

if platform.system() == 'Linux' and platform.machine() == 'x86_64':
  REQUIRED_PACKAGES.append(FAKE_REQUIRED_PACKAGES)

if collaborator_build:
  # If this is a collaborator build, then build an "installer" wheel and
  # add the collaborator packages as the only dependencies.
  REQUIRED_PACKAGES = [
      # Install the TensorFlow package built by AWS if the user is running
      # Linux on an Aarch64 machine.
      standard_or_nightly('tensorflow-cpu-aws', 'tf-nightly-cpu-aws') + '==' +
      _VERSION + ';platform_system=="Linux" and (platform_machine=="arm64" or '
      'platform_machine=="aarch64")',
      # Install the TensorFlow package built by Intel if the user is on a
      # Windows machine.
      standard_or_nightly('tensorflow-intel', 'tf-nightly-intel') + '==' +
      _VERSION + ';platform_system=="Windows"',
      # Install the TensorFlow package built by Apple if the user is running
      # macOS on an Apple Silicon machine.
      standard_or_nightly('tensorflow-macos', 'tf-nightly-macos') + '==' +
      _VERSION + ';platform_system=="Darwin" and platform_machine=="arm64"',
  ]

# Set up extra packages, which are optional sets of other Python package deps.
# E.g. "pip install tensorflow[and-cuda]" below installs the normal TF deps
# plus the CUDA libraries listed.
EXTRA_PACKAGES = {}
EXTRA_PACKAGES['and-cuda'] = [
    # TODO(nluehr): set nvidia-* versions based on build components.
    'nvidia-cuda-runtime-cu11 == 11.8.89',
    'nvidia-cublas-cu11 == 11.11.3.6',
    'nvidia-cufft-cu11 == 10.9.0.58',
    'nvidia-cudnn-cu11 == 8.7.0.84',
    'nvidia-curand-cu11 == 10.3.0.86',
    'nvidia-cusolver-cu11 == 11.4.1.48',
    'nvidia-cusparse-cu11 == 11.7.5.86',
    'nvidia-nccl-cu11 == 2.16.5',
    'nvidia-cuda-cupti-cu11 == 11.8.87',
    'nvidia-cuda-nvcc-cu11 == 11.8.89',
    'tensorrt == 8.5.3.1',
]

DOCLINES = __doc__.split('\n')

# pylint: disable=line-too-long
CONSOLE_SCRIPTS = [
    'toco_from_protos = tensorflow.lite.toco.python.toco_from_protos:main',
    'tflite_convert = tensorflow.lite.python.tflite_convert:main',
    'toco = tensorflow.lite.python.tflite_convert:main',
    'saved_model_cli = tensorflow.python.tools.saved_model_cli:main',
    (
        'import_pb_to_tensorboard ='
        ' tensorflow.python.tools.import_pb_to_tensorboard:main'
    ),
    # We need to keep the TensorBoard command, even though the console script
    # is now declared by the tensorboard pip package. If we remove the
    # TensorBoard command, pip will inappropriately remove it during install,
    # even though the command is not removed, just moved to a different wheel.
    # We exclude it anyway if building tf_nightly.
    standard_or_nightly('tensorboard = tensorboard.main:run_main', None),
    'tf_upgrade_v2 = tensorflow.tools.compatibility.tf_upgrade_v2_main:main',
    (
        'estimator_ckpt_converter ='
        ' tensorflow_estimator.python.estimator.tools.checkpoint_converter:main'
    ),
]
CONSOLE_SCRIPTS = [s for s in CONSOLE_SCRIPTS if s is not None]
# pylint: enable=line-too-long


class BinaryDistribution(Distribution):

  def has_ext_modules(self):
    return True


class InstallCommand(InstallCommandBase):
  """Override the dir where the headers go."""

  def finalize_options(self):
    ret = InstallCommandBase.finalize_options(self)  # pylint: disable=assignment-from-no-return
    self.install_headers = os.path.join(self.install_platlib, 'tensorflow',
                                        'include')
    self.install_lib = self.install_platlib
    return ret


class InstallHeaders(Command):
  """Override how headers are copied.

  The install_headers that comes with setuptools copies all files to
  the same directory. But we need the files to be in a specific directory
  hierarchy for -I <include_dir> to work correctly.
  """
  description = 'install C/C++ header files'

  user_options = [
      ('install-dir=', 'd', 'directory to install header files to'),
      ('force', 'f', 'force installation (overwrite existing files)'),
  ]

  boolean_options = ['force']

  def initialize_options(self):
    self.install_dir = None
    self.force = 0
    self.outfiles = []

  def finalize_options(self):
    self.set_undefined_options('install', ('install_headers', 'install_dir'),
                               ('force', 'force'))

  def mkdir_and_copy_file(self, header):
    install_dir = os.path.join(self.install_dir, os.path.dirname(header))
    # Get rid of some extra intervening directories so we can have fewer
    # directories for -I
    install_dir = re.sub('/google/protobuf_archive/src', '', install_dir)

    # Copy external code headers into tensorflow/include.
    # A symlink would do, but the wheel file that gets created ignores
    # symlink within the directory hierarchy.
    # NOTE(keveman): Figure out how to customize bdist_wheel package so
    # we can do the symlink.
    external_header_locations = [
        'tensorflow/include/external/eigen_archive/',
        'tensorflow/include/external/com_google_absl/',
    ]
    for location in external_header_locations:
      if location in install_dir:
        extra_dir = install_dir.replace(location, '')
        if not os.path.exists(extra_dir):
          self.mkpath(extra_dir)
        self.copy_file(header, extra_dir)

    if not os.path.exists(install_dir):
      self.mkpath(install_dir)
    return self.copy_file(header, install_dir)

  def run(self):
    hdrs = self.distribution.headers
    if not hdrs:
      return

    self.mkpath(self.install_dir)
    for header in hdrs:
      (out, _) = self.mkdir_and_copy_file(header)
      self.outfiles.append(out)

  def get_inputs(self):
    return self.distribution.headers or []

  def get_outputs(self):
    return self.outfiles


def find_files(pattern, root):
  """Return all the files matching pattern below root dir."""
  for dirpath, _, files in os.walk(root):
    for filename in fnmatch.filter(files, pattern):
      yield os.path.join(dirpath, filename)


so_lib_paths = [
    i for i in os.listdir('.')
    if os.path.isdir(i) and fnmatch.fnmatch(i, '_solib_*')
]

matches = []
for path in so_lib_paths:
  matches.extend(['../' + x for x in find_files('*', path) if '.py' not in x])

if os.name == 'nt':
  EXTENSION_NAME = 'python/_pywrap_tensorflow_internal.pyd'
else:
  EXTENSION_NAME = 'python/_pywrap_tensorflow_internal.so'

headers = (
    list(find_files('*.proto', 'tensorflow/compiler')) +
    list(find_files('*.proto', 'tensorflow/core')) +
    list(find_files('*.proto', 'tensorflow/python')) +
    list(find_files('*.proto', 'tensorflow/python/framework')) +
    list(find_files('*.proto', 'tensorflow/tsl')) +
    list(find_files('*.def', 'tensorflow/compiler')) +
    list(find_files('*.h', 'tensorflow/c')) +
    list(find_files('*.h', 'tensorflow/cc')) +
    list(find_files('*.h', 'tensorflow/compiler')) +
    list(find_files('*.h.inc', 'tensorflow/compiler')) +
    list(find_files('*.h', 'tensorflow/core')) +
    list(find_files('*.h', 'tensorflow/lite/kernels/shim')) +
    list(find_files('*.h', 'tensorflow/python')) +
    list(find_files('*.h', 'tensorflow/python/client')) +
    list(find_files('*.h', 'tensorflow/python/framework')) +
    list(find_files('*.h', 'tensorflow/stream_executor')) +
    list(find_files('*.h', 'tensorflow/compiler/xla/stream_executor')) +
    list(find_files('*.h', 'tensorflow/tsl')) +
    list(find_files('*.h', 'google/com_google_protobuf/src')) +
    list(find_files('*.inc', 'google/com_google_protobuf/src')) +
    list(find_files('*', 'third_party/eigen3')) +
    list(find_files('*', 'third_party/gpus')) +
    list(find_files('*.h', 'tensorflow/include/external/com_google_absl')) +
    list(find_files('*.inc', 'tensorflow/include/external/com_google_absl')) +
    list(find_files('*', 'tensorflow/include/external/eigen_archive')))

# Quite a lot of setup() options are different if this is a collaborator package
# build. We explicitly list the differences here, then unpack the dict as
# options at the end of the call to setup() below. For what each keyword does,
# see https://setuptools.pypa.io/en/latest/references/keywords.html.
if collaborator_build:
  collaborator_build_dependent_options = {
      'cmdclass': {},
      'distclass': None,
      'entry_points': {},
      'headers': [],
      'include_package_data': None,
      'packages': [],
      'package_data': {},
  }
else:
  collaborator_build_dependent_options = {
      'cmdclass': {
          'install_headers': InstallHeaders,
          'install': InstallCommand,
      },
      'distclass': BinaryDistribution,
      'entry_points': {
          'console_scripts': CONSOLE_SCRIPTS,
      },
      'headers': headers,
      'include_package_data': True,
      'packages': find_packages(),
      'package_data': {
          'tensorflow': [EXTENSION_NAME] + matches,
      },
  }

setup(
    name=project_name,
    version=_VERSION.replace('-', ''),
    description=DOCLINES[0],
    long_description='\n'.join(DOCLINES[2:]),
    long_description_content_type='text/markdown',
    url='https://www.tensorflow.org/',
    download_url='https://github.com/tensorflow/tensorflow/tags',
    author='Google Inc.',
    author_email='packages@tensorflow.org',
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRA_PACKAGES,
    # Add in any packaged data.
    zip_safe=False,
    # Supported Python versions
    python_requires='>=3.9',
    # PyPI package information.
    classifiers=sorted([
        'Development Status :: 5 - Production/Stable',
        # TODO(angerson) Add IFTTT when possible
        'Environment :: GPU :: NVIDIA CUDA :: 11.8',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ]),
    license='Apache 2.0',
    keywords='tensorflow tensor machine learning',
    **collaborator_build_dependent_options
)