template <typename Dtype>
void LogSumExpCUDAKernel(
    int B, int M, int N, Dtype *alpha, Dtype *beta, Dtype dx
);
void InnerNewtonCUDAKernel(
    int n_iter, float tol, int N, float *t, float eps, float lam,
    float *lognu, float *lognu_nJ, float *logKTu
);