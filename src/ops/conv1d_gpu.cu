#include "ctranslate2/ops/conv1d.h"

#include "cuda/utils.h"

//needed since apparently GetWorkspaceSize is now has context configuration https://rocm.docs.amd.com/en/latest/about/release-notes.html#miopen-3-2-0
//and that causes the workspace_size to sometimes not be set which causes significant slowdown because of fallback when finding the algo
size_t last_workspace_size = 0;

namespace ctranslate2 {
  namespace ops {

    template <Device D, typename T>
    void Conv1D::compute(const StorageView& input,
                         const StorageView& weight,
                         const StorageView* bias,
                         StorageView& output,
                         const StorageView* qscale) const {
      if (qscale)
        throw std::runtime_error("Quantization is not supported in this Conv1D implementation");

#ifndef CT2_WITH_CUDNN
      (void)input;
      (void)weight;
      (void)bias;
      (void)output;
      throw std::runtime_error("Conv1D on GPU currently requires the cuDNN library "
                               "which is not integrated in this build");

#else
      const int batch_size = input.dim(0);
      const int in_channels = input.dim(1);
      const int input_length = input.dim(2);
      const int output_length = output.dim(2);
      const int out_channels = weight.dim(0);
      const int kernel_size = weight.dim(2);

      miopenDataType_t data_type = cuda::get_cudnn_data_type(input.dtype());

      miopenTensorDescriptor_t input_desc;
      CUDNN_CHECK(miopenCreateTensorDescriptor(&input_desc));
      CUDNN_CHECK(miopenSet4dTensorDescriptor(input_desc, data_type,
                                             batch_size, in_channels, 1, input_length));

      miopenTensorDescriptor_t output_desc;
      CUDNN_CHECK(miopenCreateTensorDescriptor(&output_desc));
      CUDNN_CHECK(miopenSet4dTensorDescriptor(output_desc, data_type,
                                             batch_size, out_channels, 1, output_length));

      miopenTensorDescriptor_t weight_desc;
      CUDNN_CHECK(miopenCreateTensorDescriptor(&weight_desc));
      CUDNN_CHECK(miopenSet4dTensorDescriptor(weight_desc, data_type,
                                             out_channels, in_channels, 1, kernel_size));

      miopenConvolutionDescriptor_t conv_desc;
      CUDNN_CHECK(miopenCreateConvolutionDescriptor(&conv_desc));
      CUDNN_CHECK(miopenInitConvolutionDescriptor(conv_desc,
                                                  miopenConvolution,
                                                  /*pad_h=*/0, /*pad_w=*/_padding,
                                                  /*stride_h=*/1, /*stride_w=*/_stride,
                                                  /*dilation_h=*/1, /*dilation_w=*/_dilation
                                                  ));

      miopenHandle_t handle = cuda::get_cudnn_handle();

      miopenConvFwdAlgorithm_t algo;

      size_t workspace_size = 0;
      void* workspace = nullptr;
      CUDNN_CHECK(miopenConvolutionForwardGetWorkSpaceSize(handle,
                                                          input_desc,
                                                          weight_desc,
                                                          conv_desc,
                                                          output_desc,
                                                          &workspace_size));
      if(workspace_size <= 0)
        workspace_size = last_workspace_size;
      if (workspace_size > 0)
      {
        workspace = get_allocator<Device::CUDA>().allocate(workspace_size);
        last_workspace_size = workspace_size;
      }
      {
      miopenConvAlgoPerf_t convForwardAlgos;
      int algoCount = 1;
      CUDNN_CHECK(miopenFindConvolutionForwardAlgorithm(handle,
                                            input_desc,
                                            input.buffer(),
                                            weight_desc,
                                            weight.buffer(),
                                            conv_desc,
                                            output_desc,
                                            output.buffer(),
                                            algoCount,
                                            &algoCount,
                                            &convForwardAlgos,
                                            workspace,
                                            workspace_size,
                                            false //exhaustive_search
      ));
      if(algoCount <= 0)
        THROW_RUNTIME_ERROR("Couldn't find any forward algorithm for requested tensors.");

      algo = convForwardAlgos.fwd_algo;
      }

      float alpha = 1;
      float beta = 0;
      if (bias) {
        miopenTensorDescriptor_t bias_desc;
        CUDNN_CHECK(miopenCreateTensorDescriptor(&bias_desc));
        CUDNN_CHECK(miopenSet4dTensorDescriptor(bias_desc, data_type,
                                               1, out_channels, 1, 1));

        CUDNN_CHECK(miopenConvolutionForward(handle,
                                            &alpha,
                                            input_desc,
                                            input.buffer(),
                                            weight_desc,
                                            weight.buffer(),
                                            conv_desc,
                                            algo,
                                            &beta,
                                            output_desc,
                                            output.buffer(),
                                            workspace,
                                            workspace_size));

        CUDNN_CHECK(miopenConvolutionForwardBias(handle,
                                         &alpha,
                                         bias_desc,
                                         bias->buffer(),
                                         &beta,
                                         output_desc,
                                         output.buffer()));

        CUDNN_CHECK(miopenDestroyTensorDescriptor(bias_desc));
      } else {
        CUDNN_CHECK(miopenConvolutionForward(handle,
                                            &alpha,
                                            input_desc,
                                            input.buffer(),
                                            weight_desc,
                                            weight.buffer(),
                                            conv_desc,
                                            algo,
                                            &beta,
                                            output_desc,
                                            output.buffer(),
                                            workspace,
                                            workspace_size));
      
      }

      if (workspace)
        get_allocator<Device::CUDA>().free(workspace);

      CUDNN_CHECK(miopenDestroyConvolutionDescriptor(conv_desc));
      CUDNN_CHECK(miopenDestroyTensorDescriptor(weight_desc));
      CUDNN_CHECK(miopenDestroyTensorDescriptor(input_desc));
      CUDNN_CHECK(miopenDestroyTensorDescriptor(output_desc));
#endif
    }

#define DECLARE_IMPL(T)                                                 \
    template void                                                       \
    Conv1D::compute<Device::CUDA, T>(const StorageView& input,          \
                                     const StorageView& weight,         \
                                     const StorageView* bias,           \
                                     StorageView& output,               \
                                     const StorageView* qscale) const;

    DECLARE_IMPL(float)
    DECLARE_IMPL(float16_t)
    #if CUDA_CAN_USE_BF16_MATH
    DECLARE_IMPL(bfloat16_t)
    #endif
  }
}
