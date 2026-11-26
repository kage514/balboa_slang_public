static const int2 neighbor_offsets[12] = 
{
    // Structural
    int2(-1,  0), int2(1,  0), int2(0, -1), int2(0, 1),
    // Shear
    int2(-1, -1), int2(1, -1), int2(-1, 1), int2(1, 1),
    // Bending
    int2(-2,  0), int2(2,  0), int2(0, -2), int2(0, 2),
};

[shader("compute")]
[numthreads(64, 1, 1)]
void project_constraints(uint tid : SV_DispatchThreadID,
                         StructuredBuffer<float3> positions_in,
                         RWStructuredBuffer<float3> positions_out,
                         StructuredBuffer<float> inv_masses,
                         uniform int2 grid_size,
                         uniform float2 spacing) {
    if (tid >= grid_size.x * grid_size.y) {
        return;
    }
    
    float3 position = positions_in[tid];
    float inverse_mass = inv_masses[tid];
    if (inverse_mass == 0.0)
    {
        positions_out[tid] = position;
        return;
    }

    int width = grid_size.x;
    int2 coord = int2(int(tid) % width, int(tid) / width);
    float3 correction_sum = float3(0.0, 0.0, 0.0);
    uint neighbor_count = 0;

    for (int i = 0; i < 12; ++i)
    {
        int2 neighbor_coord = coord + neighbor_offsets[i];
        if (any(neighbor_coord < int2(0, 0)) ||
            any(neighbor_coord >= grid_size))
        {
            continue;
        }

        uint neighbor = uint(neighbor_coord.y * width + neighbor_coord.x);
        ++neighbor_count;

        float3 delta = positions_in[neighbor] - position;
        float distance = length(delta);
        if (distance > 0.0)
        {
            float rest_length = length(float2(neighbor_offsets[i]) * spacing);
            float weight_scale = inverse_mass / (inverse_mass + inv_masses[neighbor]);
            correction_sum += weight_scale * (distance - rest_length) * (delta / distance);
        }
    }

    if (neighbor_count > 0)
    {
        position += correction_sum / float(neighbor_count);
    }
    positions_out[tid] = position;
}
