//Test whether the point is inside the triangle
bool is_point_in_triangle(float2 p, float2 v0, float2 v1, float2 v2)
{
    float2 e0 = v0 - v1;
    float2 e1 = v1 - v2;
    float2 e2 = v2 - v0;

    float2 n0 = float2(e0.y, -e0.x);
    float2 n1 = float2(e1.y, -e1.x);
    float2 n2 = float2(e2.y, -e2.x);

    float s0 = dot((p - v0), n0);
    float s1 = dot((p - v1), n1);
    float s2 = dot((p - v2), n2);

    return (s0 >= 0 && s1 >= 0 && s2 >= 0) 
        || (s0 <= 0 && s1 <= 0 && s2 <= 0);
}

float calculate_area(float2 p0, float2 p1, float2 p2)
{
    float2 u = p1 - p0;
    float2 v = p2 - p0;
    return length(cross(float3(u, 0.0), float3(v, 0.0))) / 2;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void clear_buffers(uint tid : SV_DispatchThreadID,
                   uniform uint num_pixels,
                   RWStructuredBuffer<uint> depth_buffer,
                   RWStructuredBuffer<uint> triangle_id_buffer) {
    if (tid >= num_pixels)
        return;

    depth_buffer[tid] = 0xffffffff;
    triangle_id_buffer[tid] = 0xffffffff;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void transform_vertices(
    uint tid : SV_DispatchThreadID,
    uniform uint num_vertices,
    uniform uint vertex_offset,
    uniform float4x4 MVP,
    StructuredBuffer<float3> vertices,
    RWStructuredBuffer<float3> projected_vertices)
{
    if (tid >= num_vertices)
        return;

    uint i = vertex_offset + tid;
    float4 p = mul(MVP, float4(vertices[i], 1.0));
    projected_vertices[i] = p.xyz / p.w;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rasterize_depth(
        uint tid : SV_DispatchThreadID,
        uniform uint num_triangles,
        uniform uint width,
        uniform uint height,
        uniform float s,
        uniform float a,
        StructuredBuffer<uint3> faces,
        // A projected vertex store the screen space coordinates
        // in xy, and the inverse depth in z.
        StructuredBuffer<float3> projected_vertices,
        RWStructuredBuffer<uint> depth_buffer) {
    if (tid >= num_triangles)
        return;
    
    uint3 face = faces[tid];

    float3 v0 = projected_vertices[face.x];
    float3 v1 = projected_vertices[face.y];
    float3 v2 = projected_vertices[face.z];

    float2 sp0 = v0.xy;
    float2 sp1 = v1.xy;
    float2 sp2 = v2.xy;

    float inv_depth0 = v0.z;
    float inv_depth1 = v1.z;
    float inv_depth2 = v2.z;

    int2 aabb_min = int2(floor(min(min(sp0, sp1), sp2)));
    int2 aabb_max = int2(ceil(max(max(sp0, sp1), sp2)));

    aabb_min = max(aabb_min, int2(0, 0));
    aabb_max = min(aabb_max, int2(int(width) - 1, int(height) - 1));

    float area = calculate_area(sp0, sp1, sp2);
    if (area == 0.0)
        return;

    for (int y = aabb_min.y; y <= aabb_max.y; y++)
    {
        for (int x = aabb_min.x; x <= aabb_max.x; x++)
        {
            float2 pixel_center = float2(x + 0.5, y + 0.5);
            if(is_point_in_triangle(pixel_center, sp0, sp1, sp2))
            {
                float b0_prime = calculate_area(pixel_center, sp1, sp2) / area;
                float b1_prime = calculate_area(sp0, pixel_center, sp2) / area;
                float b2_prime = calculate_area(sp0, sp1, pixel_center) / area;

                float b0_inv_z = b0_prime * inv_depth0;
                float b1_inv_z = b1_prime * inv_depth1;
                float b2_inv_z = b2_prime * inv_depth2;
                
                //This is the correct b coords.
                float b0 = b0_inv_z / (b0_inv_z + b1_inv_z + b2_inv_z);
                float b1 = b1_inv_z / (b0_inv_z + b1_inv_z + b2_inv_z);
                float b2 = b2_inv_z / (b0_inv_z + b1_inv_z + b2_inv_z);

                float z_depth = b0 / inv_depth0 + b1 / inv_depth1 + b2 / inv_depth2;

                uint depth_bits = asuint(z_depth);
                uint pixel_index = uint(y) * width + uint(x);
                InterlockedMin(depth_buffer[pixel_index], depth_bits);
            }
        }
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void rasterize_id(
        uint tid : SV_DispatchThreadID,
        uniform uint num_triangles,
        uniform uint width,
        uniform uint height,
        uniform float s,
        uniform float a,
        StructuredBuffer<uint3> faces,
        StructuredBuffer<float3> projected_vertices,
        RWStructuredBuffer<uint> depth_buffer,
        RWStructuredBuffer<uint> triangle_id_buffer) {
    if (tid >= num_triangles)
        return;

    uint3 face = faces[tid];

    float3 v0 = projected_vertices[face.x];
    float3 v1 = projected_vertices[face.y];
    float3 v2 = projected_vertices[face.z];

    float2 sp0 = v0.xy;
    float2 sp1 = v1.xy;
    float2 sp2 = v2.xy;

    float inv_depth0 = v0.z;
    float inv_depth1 = v1.z;
    float inv_depth2 = v2.z;

    int2 aabb_min = int2(floor(min(min(sp0, sp1), sp2)));
    int2 aabb_max = int2(ceil(max(max(sp0, sp1), sp2)));

    aabb_min = max(aabb_min, int2(0, 0));
    aabb_max = min(aabb_max, int2(int(width) - 1, int(height) - 1));

    float area = calculate_area(sp0, sp1, sp2);
    if (area == 0.0)
        return;

    for (int y = aabb_min.y; y <= aabb_max.y; y++)
    {
        for (int x = aabb_min.x; x <= aabb_max.x; x++)
        {
            float2 pixel_center = float2(x + 0.5, y + 0.5);
            if(is_point_in_triangle(pixel_center, sp0, sp1, sp2))
            {
                float b0_prime = calculate_area(pixel_center, sp1, sp2) / area;
                float b1_prime = calculate_area(sp0, pixel_center, sp2) / area;
                float b2_prime = calculate_area(sp0, sp1, pixel_center) / area;

                float b0_inv_z = b0_prime * inv_depth0;
                float b1_inv_z = b1_prime * inv_depth1;
                float b2_inv_z = b2_prime * inv_depth2;
                
                //This is the correct b coords.
                float b0 = b0_inv_z / (b0_inv_z + b1_inv_z + b2_inv_z);
                float b1 = b1_inv_z / (b0_inv_z + b1_inv_z + b2_inv_z);
                float b2 = b2_inv_z / (b0_inv_z + b1_inv_z + b2_inv_z);

                float z_depth = b0 / inv_depth0 + b1 / inv_depth1 + b2 / inv_depth2;

                uint pixel_index = uint(y) * width + uint(x);
                if(asuint(z_depth) == depth_buffer[pixel_index])
                {
                    InterlockedMin(triangle_id_buffer[pixel_index], tid);
                }
            }
        }
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void shade(
        uint2 tid : SV_DispatchThreadID,
        uniform uint width,
        uniform uint height,
        uniform float s,
        uniform float a,
        StructuredBuffer<uint3> faces,
        StructuredBuffer<float3> vertex_colors,
        StructuredBuffer<float3> projected_vertices,
        StructuredBuffer<uint> triangle_id_buffer,
        RWTexture2D<float4> image) {
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    uint pixel_index = uint(tid.y) * width + uint(tid.x);
    uint face_id = triangle_id_buffer[pixel_index];
    if(face_id != 0xffffffff)
    {
        uint3 face = faces[face_id];

        float3 v0 = projected_vertices[face.x];
        float3 v1 = projected_vertices[face.y];
        float3 v2 = projected_vertices[face.z];

        float3 c0 = vertex_colors[face.x];
        float3 c1 = vertex_colors[face.y];
        float3 c2 = vertex_colors[face.z];

        float2 sp0 = v0.xy;
        float2 sp1 = v1.xy;
        float2 sp2 = v2.xy;

        float inv_depth0 = v0.z;
        float inv_depth1 = v1.z;
        float inv_depth2 = v2.z;

        float area = calculate_area(sp0, sp1, sp2);
        if (area == 0.0)
            return;

        float2 pixel_center = float2(tid.x + 0.5, tid.y + 0.5);
        float b0_prime = calculate_area(pixel_center, sp1, sp2) / area;
        float b1_prime = calculate_area(sp0, pixel_center, sp2) / area;
        float b2_prime = calculate_area(sp0, sp1, pixel_center) / area;

        float b0_inv_z = b0_prime * inv_depth0;
        float b1_inv_z = b1_prime * inv_depth1;
        float b2_inv_z = b2_prime * inv_depth2;
        
        //This is the correct b coords.
        float b0 = b0_inv_z / (b0_inv_z + b1_inv_z + b2_inv_z);
        float b1 = b1_inv_z / (b0_inv_z + b1_inv_z + b2_inv_z);
        float b2 = b2_inv_z / (b0_inv_z + b1_inv_z + b2_inv_z);

        float3 color = b0 * c0 + b1 * c1 + b2 * c2;
        image[tid] = float4(color, 1.0);
    }
}
