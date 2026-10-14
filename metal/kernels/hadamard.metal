// One threadgroup per block/row. All butterflies use FP32, including the prompt's FP16 inputs/outputs.
struct HadArgs { uint width, rows, block, indirect; };
template <typename T>
kernel void hadamard(device const T* x [[buffer(0)]], device T* y [[buffer(1)]],
                     device const float* signs [[buffer(2)]], constant HadArgs& a [[buffer(3)]],
                     device const uint* meta [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
                     ushort tid [[thread_index_in_threadgroup]], ushort3 shape [[threads_per_threadgroup]]) {
    const ushort nth = shape.x;
    const uint rows = a.indirect ? meta[1] : a.rows;
    if (tg.y >= rows) return;
    threadgroup float v[4096];
    const uint start = tg.y * a.width + tg.x * a.block;
    for (uint i = tid; i < a.block; i += nth) v[i] = float(x[start + i]) * signs[tg.x * a.block + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint len = 1; len < a.block; len <<= 1) {
        for (uint pair = tid; pair < a.block / 2; pair += nth) {
            const uint j = (pair / len) * (2 * len) + pair % len;
            const float p = v[j], q = v[j + len];
            v[j] = p + q; v[j + len] = p - q;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float scale = 1.0f / sqrt(float(a.block));
    for (uint i = tid; i < a.block; i += nth) y[start + i] = T(v[i] * scale);
}
template [[host_name("hadamard_f32")]] kernel decltype(hadamard<float>) hadamard<float>;
template [[host_name("hadamard_f16")]] kernel decltype(hadamard<half>) hadamard<half>;
