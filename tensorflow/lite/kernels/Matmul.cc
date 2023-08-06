#include <stdint.h>
#include "/home/hy/tensorflow_1/tensorflow/lite/core/c/common.h"
#include "/home/hy/tensorflow_1/tensorflow/lite/kernels/internal/tensor.h"
#include "/home/hy/tensorflow_1/tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "/home/hy/tensorflow_1/tensorflow/lite/kernels/kernel_util.h"
namespace tflite {
namespace ops {
namespace builtin {
namespace matmul {

constexpr int kInputATensor = 0;
constexpr int kInputBTensor = 1;
constexpr int kOutputTensor = 0;
//Prepare函数
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  //确保输入的数量符合
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  //确保获取输入张量 input_a 和 input_b 的操作执行成功
  const TfLiteTensor* input_a;
  const TfLiteTensor* input_b;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputATensor, &input_a));
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputBTensor, &input_b));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kOutputTensor, &output));
  //确保维度符合
  TF_LITE_ENSURE_EQ(context, input_a->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, input_b->dims->size, 2);
  TF_LITE_ENSURE_EQ(context, input_a->dims->data[1], input_b->dims->data[0]);
  //调整输出张量的大小，以确保它与输入张量的维度相匹配。
  TfLiteIntArray* output_size = TfLiteIntArrayCreate(2);
  output_size->data[0] = input_a->dims->data[0];
  output_size->data[1] = input_b->dims->data[1];
  TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, output, output_size));
  //保输出张量 output 的大小与输入张量input_a和input_b的大小相匹配。
  TF_LITE_ENSURE(context, NumDimensions(output) == 2);
  TF_LITE_ENSURE(context, output->dims->data[0] == input_a->dims->data[0]);
  TF_LITE_ENSURE(context, output->dims->data[1] == input_b->dims->data[1]);
  //将输出张量的类型设置为与输入张量 input_a 的类型相同
  output->type = input_a->type;

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  //确保获取输入张量 input_a 和 input_b 的操作执行成功,获取output张量
  const TfLiteTensor* input_a;
  const TfLiteTensor* input_b;
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputATensor, &input_a));
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputBTensor, &input_b));
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kOutputTensor, &output));
  //获取输入张量和输出张量的数据指针
  const int32_t* a_data = GetTensorData<int32_t>(input_a);
  const int32_t* b_data = GetTensorData<int32_t>(input_b);
  int32_t* output_data = GetTensorData<int32_t>(output);
  //获取输入张量input_a的行数（rows_a）,输入张量input_a的列数和input_b的行数（cols_a）,输入张量 input_b 的列数（cols_b）
  const int rows_a = input_a->dims->data[0];
  const int cols_a = input_a->dims->data[1];
  const int cols_b = input_b->dims->data[1];
  //O(N^3)矩阵乘法运算
  for (int i=0;i<rows_a;i++) {
    for (int j=0;j<cols_b;j++) {
      output_data[i*cols_b+j] = 0;
      for (int k=0;k<cols_a;k++) {
        output_data[i*cols_b+j] += a_data[i*cols_a+k] * b_data[k*cols_b+j];
      }
    }
  }

  return kTfLiteOk;
}


// 注册矩阵乘法操作函数
TfLiteRegistration* Register_MATRIX_MULTIPLY() {
  static TfLiteRegistration r = {nullptr, nullptr, Prepare,
                                 Eval};
  return &r;
}  // namespace matmul
}
}  // namespace builtin
}  // namespace ops
}  // namespace tflite
