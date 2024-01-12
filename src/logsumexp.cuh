template <typename Dtype>
void LogSumExpCUDAKernel(
    int B, int M, int N, Dtype *alpha, Dtype *beta, Dtype dx
);

template <typename Dtype>
void LogSumExpCUDAKernel_xyeps(
    int B, int M, int N, Dtype *alpha, Dtype *beta, 
    Dtype dx, Dtype dy, Dtype eps
);