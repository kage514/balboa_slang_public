[shader("compute")]
[numthreads(64, 1, 1)]
void project_constraints(uint tid : SV_DispatchThreadID,
                         StructuredBuffer<float3> positions_in,
                         RWStructuredBuffer<float3> positions_out,
                         StructuredBuffer<float> inv_masses,
                         // The CSR matrix data structure
                         StructuredBuffer<uint> row_ptr,
                         StructuredBuffer<uint> neighbors,
                         // Rest lengthes of the neighbor edges
                         StructuredBuffer<float> neighbor_rest,
                         uniform uint num_particles) {
    if (tid >= num_particles) {
        return;
    }

    float3 position = positions_in[tid];
    float inverse_mass = inv_masses[tid];
    if (inverse_mass == 0.0)
    {
        positions_out[tid] = position;
        return;
    }

    uint first_slot = row_ptr[tid];
    uint end_slot = row_ptr[tid + 1];
    uint neighbor_count = end_slot - first_slot;
    float3 correction_sum = float3(0.0, 0.0, 0.0);

    for (uint i = first_slot; i < end_slot; ++i)
    {
        uint neighbor = neighbors[i];
        float3 delta = positions_in[neighbor] - position;
        float distance = length(delta);

        if (distance > 0.0)
        {
            float rest_length = neighbor_rest[i];
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
