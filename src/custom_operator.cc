#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#undef ORT_API_MANUAL_INIT

#include "onnxruntime_lite_custom_op.h"

#include <cmath>
#include <mutex>
#include <system_error>
#include <vector>

void KernelOne(const Ort::Custom::Tensor<float> &X,
               const Ort::Custom::Tensor<float> &Y,
               Ort::Custom::Tensor<float> &Z) {
  auto input_shape = X.Shape();
  auto x_raw = X.Data();
  auto y_raw = Y.Data();
  auto z_raw = Z.Allocate(input_shape);
  for (int64_t i = 0; i < Z.NumberOfElement(); ++i) {
    z_raw[i] = x_raw[i] + y_raw[i];
  }
}

static const char *c_OpDomain = "test.customop";

static void AddOrtCustomOpDomainToContainer(Ort::CustomOpDomain &&domain) {
  static std::vector<Ort::CustomOpDomain> ort_custom_op_domain_container;
  static std::mutex ort_custom_op_domain_mutex;
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  ort_custom_op_domain_container.push_back(std::move(domain));
}

OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options,
                                          const OrtApiBase *api) {

  Ort::CustomOpDomain domain{c_OpDomain};
  Ort::CustomOpDomain domain_v2{"v2"};

  Ort::UnownedSessionOptions session_options(options);
  session_options.Add(domain);
  session_options.Add(domain_v2);
  AddOrtCustomOpDomainToContainer(std::move(domain));
  AddOrtCustomOpDomainToContainer(std::move(domain_v2));
}

OrtStatus *ORT_API_CALL RegisterCustomOpsAltName(OrtSessionOptions *options,
                                                 const OrtApiBase *api) {
  return RegisterCustomOps(options, api);
}

int main() {
  Ort::CustomOpDomain v1_domain{"v1"};
  // please make sure that custom_op_one has the same lifetime as the consuming
  // session
  std::unique_ptr<Ort::Custom::OrtLiteCustomOp> custom_op_one{
      Ort::Custom::CreateLiteCustomOp("CustomOpOne", "CPUExecutionProvider",
                                      KernelOne)};
  v1_domain.Add(custom_op_one.get());
  Ort::SessionOptions session_options;
  session_options.Add(v1_domain);
  // create a session with the session_options ...
}
