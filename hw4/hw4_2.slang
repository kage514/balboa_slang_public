[shader("compute")]
[numthreads(64, 1, 1)]
void predict(uint tid : SV_DispatchThreadID,
             // Updated position
             RWStructuredBuffer<float3> positions,
             // Store the position before update
             RWStructuredBuffer<float3> old_positions,
             RWStructuredBuffer<float3> velocities,
             StructuredBuffer<float> inv_masses,
             uniform uint num_particles,
             // time step size
             uniform float dt) {
    if (tid >= num_particles) {
        return;
    }

    float3 position = positions[tid];
    old_positions[tid] = position;

    //fixed point if inv_mass == 0.0
    if (inv_masses[tid] == 0.0)
    {
        return;
    }

    const float3 gravity = float3(0.0, -9.8, 0.0);
    float3 velocity = velocities[tid] + gravity * dt;

    velocities[tid] = velocity;
    positions[tid] = position + velocity * dt;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void project_constraints(uint tid : SV_DispatchThreadID,
                         StructuredBuffer<float3> positions_in,
                         RWStructuredBuffer<float3> positions_out,
                         StructuredBuffer<float> inv_masses,
                         StructuredBuffer<float> rest_lengths,
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

    uint first_edge = tid > 0 ? tid - 1 : 0;
    uint end_edge = min(tid + 1, num_particles - 1);
    float3 correction_sum = float3(0.0, 0.0, 0.0);

    //loop from first_edge to end_edge - 1, end_edge is not included.
    for (uint edge = first_edge; edge < end_edge; ++edge)
    {
        uint neighbor = edge == tid ? edge + 1 : edge; //use edge for left half, edge + 1 for right half.
        float3 delta = positions_in[neighbor] - position;
        float distance = length(delta);

        if (distance > 0.0)
        {
            float3 direction = delta / distance;
            float loss = distance - rest_lengths[edge];
            float weight_scale = inverse_mass / (inverse_mass + inv_masses[neighbor]);
            correction_sum += weight_scale * loss * direction;
        }
    }

    uint neighbor_count = end_edge - first_edge;
    if (neighbor_count > 0)
    {
        position += correction_sum / float(neighbor_count);
    }

    positions_out[tid] = position;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void update_velocity(uint tid : SV_DispatchThreadID,
                     RWStructuredBuffer<float3> positions,
                     StructuredBuffer<float3> old_positions,
                     RWStructuredBuffer<float3> velocities,
                     StructuredBuffer<float> inv_masses,
                     uniform uint num_particles,
                     // time step size
                     uniform float dt) {
    if (tid >= num_particles) {
        return;
    }
        
    if (inv_masses[tid] == 0.0)
    {
        return;
    }   

    const float floor_y = -0.97;
    float3 position = positions[tid];
    position.y = max(position.y, floor_y);

    positions[tid] = position;
    velocities[tid] = (position - old_positions[tid]) / dt;
}
