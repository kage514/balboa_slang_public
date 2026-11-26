[shader("compute")]
[numthreads(64, 1, 1)]
void detect_collision(uint tid : SV_DispatchThreadID,
                      StructuredBuffer<float3> positions,
                      StructuredBuffer<float> inv_masses,
                      // Plane stores (a, b, c, d) where ax+by+cz=d
                      // a, b, c are normalized to a unit 3D vector
                      StructuredBuffer<float4> planes,
                      // Sphere stores center (xyz) + radius (w)
                      StructuredBuffer<float4> spheres,
                      // Contact plane: dot(n, x) = d, with unit normal n
                      // Set to (0, 0, 0, 0) if there is no contact
                      RWStructuredBuffer<float4> contact_planes,
                      uniform uint num_particles,
                      uniform uint num_planes,
                      uniform uint num_spheres,
                      // How much should we shift the generated 
                      // contact plane in the normal direction
                      // (similar to how we shifted the floor by 0.03
                      //  earlier)
                      uniform float collision_distance,
                      // When the signed distance function
                      // evaluates to a value below 
                      // contact_margin + collision_distance
                      // (even when it's positive), we count the
                      // particle to be inside the collider and
                      // a collision event has happened.
                      uniform float contact_margin) {
    if (tid >= num_particles) {
        return;
    }
    
    float4 contact_plane = float4(0.0, 0.0, 0.0, 0.0);
    if (inv_masses[tid] == 0.0)
    {
        contact_planes[tid] = contact_plane;
        return;
    }

    float3 position = positions[tid];
    float min_distance = contact_margin + collision_distance;

    for (uint i = 0; i < num_planes; ++i)
    {
        float4 plane = planes[i];
        float distance = dot(plane.xyz, position) - plane.w;
        if (distance < min_distance)
        {
            min_distance = distance;
            contact_plane = float4(plane.xyz, plane.w + collision_distance);
        }
    }

    for (uint i = 0; i < num_spheres; ++i)
    {
        float4 sphere = spheres[i];
        float3 delta = position - sphere.xyz;
        float center_distance = length(delta);
        float distance = center_distance - sphere.w;
        if (distance < min_distance) {
            min_distance = distance;

            float3 normal = float3(0.0, 1.0, 0.0);
            if (center_distance > 0.0)
            {
                normal = delta / center_distance;
            }

            float d = dot(normal, sphere.xyz) + sphere.w;
            contact_plane = float4(normal, d + collision_distance);
        }
    }

    contact_planes[tid] = contact_plane;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void project_constraints(uint tid : SV_DispatchThreadID,
                         StructuredBuffer<float3> positions_in,
                         StructuredBuffer<float3> old_positions,
                         RWStructuredBuffer<float3> positions_out,
                         StructuredBuffer<float> inv_masses,
                         StructuredBuffer<uint> row_ptr,
                         StructuredBuffer<uint> neighbors,
                         StructuredBuffer<float> neighbor_rest,
                         // Contact plane: dot(n, x) = d, with unit normal n
                         StructuredBuffer<float4> contact_planes,
                         uniform uint num_particles,
                         uniform float static_friction,
                         uniform float kinetic_friction) {
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

        if (distance > 0.0) {
            float rest_length = neighbor_rest[i];
            float weight_scale = inverse_mass / (inverse_mass + inv_masses[neighbor]);
            correction_sum += weight_scale * (distance - rest_length) * (delta / distance);
        }
    }

    if (neighbor_count > 0)
    {
        position += correction_sum / float(neighbor_count);
    }

    float4 contact_plane = contact_planes[tid];
    float3 normal = contact_plane.xyz;
    if (any(normal != float3(0.0, 0.0, 0.0)))
    {
        float penetration = contact_plane.w - dot(normal, position);
        if (penetration > 0.0)
        {
            position += penetration * normal;

            float3 displacement = position - old_positions[tid];
            float3 tangent = displacement - dot(displacement, normal) * normal;
            float tangent_length = length(tangent);

            if (tangent_length <= static_friction * penetration)
            {
                position -= tangent;
            }
            else if (tangent_length > 0.0)
            {
                position -= kinetic_friction * penetration * (tangent / tangent_length);
            }
        }
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
                     uniform float dt) {
    if (tid >= num_particles) {
        return;
    }
    
    if (inv_masses[tid] == 0.0)
    {
        return;
    }

    float3 position = positions[tid];
    velocities[tid] = (position - old_positions[tid]) / dt;
}
