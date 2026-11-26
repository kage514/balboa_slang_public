[shader("compute")]
[numthreads(64, 1, 1)]
void compute_main(uint tid : SV_DispatchThreadID,
                  RWStructuredBuffer<float3> positions,
                  RWStructuredBuffer<float3> velocities,
                  uniform uint num_particles,
                  // time step size
                  uniform float dt) {
    if (tid >= num_particles) {
        return;
    }

    float3 position = positions[tid];
    float3 velocity = velocities[tid];
    const float3 gravity = float3(0.0, -9.8, 0.0);
    const float floor_y = -0.97;

    velocity += gravity * dt; //Change in V = a * t
    position += velocity * dt; //Displacement = V * t

    if (position.y < floor_y)
    {
        position.y = floor_y;
        velocity.y = -velocity.y;
    }

    positions[tid] = position;
    velocities[tid] = velocity;
}
