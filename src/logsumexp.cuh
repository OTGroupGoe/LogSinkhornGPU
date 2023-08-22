template <typename Dtype>
void LogSumExpCUDAKernel(
    int B, int M, int N, Dtype *alpha, Dtype *beta, Dtype dx
);