template <typename Dtype>
void BalanceKernel(
    int B, int C, int N, Dtype *nu_basic, Dtype *mass_delta, Dtype thresh_step
);