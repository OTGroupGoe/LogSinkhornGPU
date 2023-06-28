void LogSumExpCUDAKernel(
    int B, int M, int N, float *alpha, float *beta, float dx
);
void InnerNewtonCUDAKernel(
    int n_iter, float tol, int N, float *t, float eps, float lam,
    float *lognu, float *lognu_nJ, float *logKTu
);