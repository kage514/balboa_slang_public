struct BVHNode {
    float3 bbox_min;
    float3 bbox_max;

    // which BVH node we should skip
    // to if this node is totally not
    // intersecting the ray
    // if next_index < 0, then we don't
    // have the next node to skip to
    int next_index;

    uint triangle_begin;
    uint triangle_end;

    bool is_leaf;
};

float point_aabb_distance(float3 x, float3 bmin, float3 bmax) {
    return length(x - clamp(x, bmin, bmax));
}

struct ClosestPoint {
    float3 position;
    float3 bary;
};

// From https://iquilezles.org/articles/distfunctions/
ClosestPoint closest_point_triangle(
        float3 p, float3 a, float3 b, float3 c) {
    float3 ba = b - a, pa = p - a;
    float3 cb = c - b, pb = p - b;
    float3 ac = a - c, pc = p - c;
    float3 nor = cross(ba, ac);

    float t0 = clamp(dot(ba, pa) / dot(ba, ba), 0.0, 1.0);
    float t1 = clamp(dot(cb, pb) / dot(cb, cb), 0.0, 1.0);
    float t2 = clamp(dot(ac, pc) / dot(ac, ac), 0.0, 1.0);

    float3 q0 = a + ba * t0;
    float3 q1 = b + cb * t1;
    float3 q2 = c + ac * t2;

    if (sign(dot(cross(ba, nor), pa)) +
        sign(dot(cross(cb, nor), pb)) +
        sign(dot(cross(ac, nor), pc)) < 2.0) {

        float d0 = dot(p - q0, p - q0);
        float d1 = dot(p - q1, p - q1);
        float d2 = dot(p - q2, p - q2);

        if (d0 < d1 && d0 < d2)
            return {q0, float3(1.0 - t0, t0, 0.0)};
        if (d1 < d2)
            return {q1, float3(0.0, 1.0 - t1, t1)};
        return {q2, float3(t2, 0.0, 1.0 - t2)};
    }

    float inv_nor2 = 1.0 / dot(nor, nor);
    float3 q = p - nor * (dot(nor, pa) * inv_nor2);

    float wb = dot(cross(pa, ac), nor) * inv_nor2;
    float wc = dot(cross(pa, ba), nor) * inv_nor2;
    return {q, float3(1.0 - wb - wc, wb, wc)};
}


[shader("compute")]
[numthreads(64, 1, 1)]
void detect_collision(uint tid : SV_DispatchThreadID,
                      StructuredBuffer<float3> positions,
                      StructuredBuffer<float> inv_masses,
                      StructuredBuffer<float4> planes,
                      StructuredBuffer<float4> spheres,
                      StructuredBuffer<float3> mesh_vertices,
                      StructuredBuffer<float3> mesh_vertex_normals,
                      StructuredBuffer<uint3> mesh_triangles,
                      StructuredBuffer<BVHNode> bvh_nodes,
                      RWStructuredBuffer<float4> contact_planes,
                      uniform uint num_particles,
                      uniform uint num_planes,
                      uniform uint num_spheres,
                      uniform uint num_bvh_nodes,
                      uniform float collision_distance,
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
        if (distance < min_distance)
        {
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

    // Find the closest mesh triangle by unsigned distance.
    float mesh_min_distance = contact_margin + collision_distance;
    const uint invalid_triangle = 0xffffffffu;
    uint closest_triangle = invalid_triangle;
    ClosestPoint closest = { float3(0.0, 0.0, 0.0), float3(0.0, 0.0, 0.0) };
    float3 closest_face_normal = float3(0.0, 0.0, 0.0);

    int node_index = 0;
    while (node_index >= 0 && uint(node_index) < num_bvh_nodes)
    {
        BVHNode node = bvh_nodes[node_index];
        float box_distance = point_aabb_distance(position, node.bbox_min, node.bbox_max);

        if (box_distance > mesh_min_distance)
        {
            node_index = node.next_index;
            continue;
        }

        if (!node.is_leaf)
        {
            ++node_index;
            continue;
        }

        for (uint i = node.triangle_begin; i < node.triangle_end; ++i)
        {
            uint3 triangle = mesh_triangles[i];
            float3 p0 = mesh_vertices[triangle.x];
            float3 p1 = mesh_vertices[triangle.y];
            float3 p2 = mesh_vertices[triangle.z];

            float3 face_normal = cross(p1 - p0, p2 - p0);

            ClosestPoint candidate = closest_point_triangle(position, p0, p1, p2);
            float distance = length(position - candidate.position);
            if (distance < mesh_min_distance)
            {
                mesh_min_distance = distance;
                closest_triangle = i;
                closest = candidate;
                closest_face_normal = face_normal;
            }
        }

        node_index = node.next_index;
    }

    if (closest_triangle != invalid_triangle)
    {
        uint3 triangle = mesh_triangles[closest_triangle];
        float3 vertex_normal =
            closest.bary.x * mesh_vertex_normals[triangle.x]
          + closest.bary.y * mesh_vertex_normals[triangle.y]
          + closest.bary.z * mesh_vertex_normals[triangle.z];

        float3 normal = normalize(closest_face_normal);
        if (dot(normal, vertex_normal) < 0.0)
        {
            normal = -normal;
        }

        float d = dot(normal, closest.position);
        float distance = dot(normal, position) - d;
        if (distance < min_distance)
        {
            contact_plane = float4(normal, d + collision_distance);
        }
    }

    contact_planes[tid] = contact_plane;
}