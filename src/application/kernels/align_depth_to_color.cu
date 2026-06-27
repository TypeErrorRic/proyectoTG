extern "C" __global__
void align_depth_to_color(
    const float* __restrict__ depth,    // (Hd*Wd)
    const float* __restrict__ A,        // (Hd*Wd*3) precomputado: R_cd * ray_d
    const float* __restrict__ t,        // (3) traslacion depth->color
    const float fx, const float fy,
    const float cx, const float cy,
    const float max_depth,              // <=0: desactivado
    const int Hd, const int Wd,
    const int Hc, const int Wc,
    unsigned int* __restrict__ out_bits // (Hc*Wc) inicializado a +inf
){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int N = Hd * Wd;
    if (idx >= N) return;

    float z = depth[idx];
    if (!(z > 0.0f) || !isfinite(z)) return;
    if (max_depth > 0.0f && z > max_depth) return;

    // A indexado linealmente (x3)
    int aidx = idx * 3;
    float ax = A[aidx + 0];
    float ay = A[aidx + 1];
    float az = A[aidx + 2];

    // Transformar punto de depth->color
    float Xcx = ax * z + t[0];
    float Xcy = ay * z + t[1];
    float Xcz = az * z + t[2];
    if (!(Xcz > 0.0f) || !isfinite(Xcz)) return;

    // Proyectar a pixel de color
    float u = fx * (Xcx / Xcz) + cx;
    float v = fy * (Xcy / Xcz) + cy;

    int ui = (int)roundf(u);
    int vi = (int)roundf(v);
    if (ui < 0 || ui >= Wc || vi < 0 || vi >= Hc) return;

    // Z-buffer: quedarse con el punto mas cercano (menor Z en camara color)
    unsigned int zbits = __float_as_uint(Xcz);
    int o = vi * Wc + ui;
    atomicMin(&out_bits[o], zbits);
}
