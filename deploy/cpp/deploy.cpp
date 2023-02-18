#include <torch/script.h> // One-stop header.
#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }


  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";

  std::cout<<"Running Model..."<<std::endl;//DEBUGGING
  module.eval();
  int nnodes = 10; //NOTE: CAN CHANGE
  int nfeatures = 6; //NOTE: DO NOT CHANGE

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones({nnodes,nfeatures}));
  inputs.push_back(torch::randint(nnodes,{2,nnodes}));

  // Execute the model and turn its output into a tensor.
  at::Tensor output = module.forward(inputs).toTensor();
  std::cout << "output = " << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
}
