load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow/compiler/xla/python/ifrt:friends"],
    licenses = ["notice"],
)

gentbl_cc_library(
    name = "passes_inc_gen",
    tbl_outs = [
        (
            [
                "-gen-pass-decls",
                "-name=IfrtIr",
            ],
            "passes.h.inc",
        ),
    ],
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "passes.td",
    deps = [
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
)

cc_library(
    name = "passes",
    srcs = [
        "ifrt_duplicated_callee_elimination_pass.cc",
        "spmd_expandable_interface_verification_pass.cc",
        "spmd_expansion_pass.cc",
    ],
    hdrs = ["passes.h"],
    deps = [
        ":constants",
        ":passes_inc_gen",
        "//tensorflow/compiler/xla/python/ifrt/ir",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/types:span",
        "@llvm-project//llvm:Support",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Support",
    ],
)

cc_library(
    name = "built_in_spmd_expansions",
    srcs = ["built_in_spmd_expansions.cc"],
    hdrs = ["built_in_spmd_expansions.h"],
    deps = [
        "//tensorflow/compiler/xla/python/ifrt/ir/transforms/spmd_expanders:spmd_expander",
        "@llvm-project//mlir:FuncDialect",
        "@llvm-project//mlir:IR",
    ],
)

cc_library(
    name = "constants",
    hdrs = ["constants.h"],
    deps = ["@llvm-project//llvm:Support"],
)