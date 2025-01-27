// custom_op_library.cc

#include "custom_operator.h"

// Custom operator libraries are not typically linked with ONNX Runtime.
// Therefore, must define ORT_API_MANUAL_INIT before including
// onnxruntime_cxx_api.h to indicate that the OrtApi object will be initialized
// manually.
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include <cmath>
#include <mutex>
#include <vector>

struct MyCustomKernel {
  MyCustomKernel(const OrtApi &api, const OrtKernelInfo *info) {}

  void Compute(OrtKernelContext *context) {
    // Setup inputs
    Ort::KernelContext ctx(context);
    Ort::ConstValue input_X = ctx.GetInput(0);
    Ort::ConstValue input_Y = ctx.GetInput(1);
    const float *X = input_X.GetTensorData<float>();
    const float *Y = input_Y.GetTensorData<float>();

    // Setup output, which is assumed to have the same dimensions as the inputs.
    std::vector<int64_t> dimensions =
        input_X.GetTensorTypeAndShapeInfo().GetShape();

    Ort::UnownedValue output = ctx.GetOutput(0, dimensions);
    float *out = output.GetTensorMutableData<float>();

    const size_t size = output.GetTensorTypeAndShapeInfo().GetElementCount();

    // Do computation
    for (size_t i = 0; i < size; i++) {
      out[i] = X[i] + Y[i];
    }
  }
};

struct MyCustomOp : Ort::CustomOpBase<MyCustomOp, MyCustomKernel> {
  void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const {
    return std::make_unique<MyCustomKernel>(api, info).release();
  };

  // Returns the name of the custom operator.
  const char *GetName() const { return "awml_pred:TRTKnnBatchMlogK(-1)"; };

  // Returns the custom operator's execution provider.
  const char *GetExecutionProviderType() const {
    return "GPUExecutionProvider";
  };

  // Returns the number of inputs.
  size_t GetInputTypeCount() const { return 2; };

  // Returns the type of each input. Both inputs are tensor(float).
  ONNXTensorElementDataType GetInputType(size_t index) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };

  // Returns the number of outputs.
  size_t GetOutputTypeCount() const { return 1; };

  // Returns the type of each output. The single output is a tensor(float).
  ONNXTensorElementDataType GetOutputType(size_t index) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };
};

// This function shows one way of keeping domains alive until the library is
// unloaded.
static void AddOrtCustomOpDomainToContainer(Ort::CustomOpDomain &&domain) {
  static std::vector<Ort::CustomOpDomain> ort_custom_op_domain_container;
  static std::mutex ort_custom_op_domain_mutex;
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  ort_custom_op_domain_container.push_back(std::move(domain));
}

// Called by ONNX Runtime to register the library's custom operators with the
// provided session options.
OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options,
                                          const OrtApiBase *api) {
  Ort::InitApi(
      api->GetApi(ORT_API_VERSION)); // Manually initialize the OrtApi to enable
                                     // use of C++ API classes and functions.

  // Custom operators are static to ensure they remain valid until the library
  // is unloaded.
  static const MyCustomOp my_custom_op;

  OrtStatus *result = nullptr;

  try {
    Ort::CustomOpDomain domain{"awml_pred"};
    domain.Add(&my_custom_op);

    Ort::UnownedSessionOptions session_options(options);
    session_options.Add(domain);
    AddOrtCustomOpDomainToContainer(std::move(domain));
  } catch (const std::exception &e) {
    Ort::Status status{e};
    result = status.release();
  }
  return result;
}
