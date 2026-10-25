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

float2 screen_to_projected(float2 p, float width, float height, float s, float a)
{
    return float2(
        (p.x / width * 2.0 - 1.0) * s * a,
        (1.0 - p.y / height * 2.0) * s
    );
}

float calculate_area(float2 p0, float2 p1, float2 p2)
{
    float2 u = p1 - p0;
    float2 v = p2 - p0;
    return length(cross(float3(u, 0.0), float3(v, 0.0))) / 2;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float2 sp0, // vertex at screen space
                  uniform float2 sp1,
                  uniform float2 sp2,
                  uniform float inv_depth0, // inverse depth in camera space
                  uniform float inv_depth1,
                  uniform float inv_depth2,
                  uniform int2 aabb_min,
                  uniform float3 c0, // vertex color
                  uniform float3 c1,
                  uniform float3 c2,
                  uniform float s,
                  uniform float a,
                  RWTexture2D<float4> color_image,
                  RWTexture2D<float> depth_image) {
    int2 ipixel = int2(tid) + int2(aabb_min);
    uint width, height;
    color_image.GetDimensions(width, height);
    if (ipixel.x < 0 || ipixel.y < 0 || ipixel.x >= width || ipixel.y >= height) {
        return;
    }

    int2 displaced_pixel_coord = aabb_min + tid;
    float2 pixel_center = displaced_pixel_coord + float2(0.5, 0.5);

    //if the point is inside the triangle, then do rendering stuff.
    if(is_point_in_triangle(pixel_center, sp0, sp1, sp2))
    {
        float2 projected_p = screen_to_projected(pixel_center, width, height, s, a);
        float2 projected_v0 = screen_to_projected(sp0, width, height, s, a);
        float2 projected_v1 = screen_to_projected(sp1, width, height, s, a);
        float2 projected_v2 = screen_to_projected(sp2, width, height, s, a);

        if (calculate_area(projected_v0, projected_v1, projected_v2) == 0.0)
        {
            return;
        }

        float b0_prime = calculate_area(projected_p, projected_v1, projected_v2) 
                            / calculate_area(projected_v0, projected_v1, projected_v2);
        float b1_prime = calculate_area(projected_v0, projected_p, projected_v2) 
                            / calculate_area(projected_v0, projected_v1, projected_v2);
        float b2_prime = calculate_area(projected_v0, projected_v1, projected_p) 
                            / calculate_area(projected_v0, projected_v1, projected_v2);
        
        float b0_inv_z = b0_prime * inv_depth0;
        float b1_inv_z = b1_prime * inv_depth1;
        float b2_inv_z = b2_prime * inv_depth2;
        
        //This is the correct b coords.
        float b0 = b0_inv_z / (b0_inv_z + b1_inv_z + b2_inv_z);
        float b1 = b1_inv_z / (b0_inv_z + b1_inv_z + b2_inv_z);
        float b2 = b2_inv_z / (b0_inv_z + b1_inv_z + b2_inv_z);

        float z_depth = b0 / inv_depth0 + b1 / inv_depth1 + b2 / inv_depth2;
        if(z_depth < depth_image[displaced_pixel_coord])
        {
            depth_image[displaced_pixel_coord] = z_depth;

            float3 interpolated_color = b0 * c0 + b1 * c1 + b2 * c2;
            color_image[displaced_pixel_coord] = float4(interpolated_color, 1.0);
        }
    }
}
