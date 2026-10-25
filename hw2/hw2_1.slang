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

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float2 sp0,
                  uniform float2 sp1,
                  uniform float2 sp2,
                  uniform int2 aabb_min,
                  uniform float3 color,
                  RWTexture2D<float4> image) {
    int2 ipixel = int2(tid) + int2(aabb_min);
    uint width, height;
    image.GetDimensions(width, height);
    if (ipixel.x < 0 || ipixel.y < 0 || ipixel.x >= width || ipixel.y >= height) {
        return;
    }

    int2 displaced_pixel_coord = aabb_min + tid;
    float2 pixel_center = displaced_pixel_coord + float2(0.5, 0.5);
    if(is_point_in_triangle(pixel_center, sp0, sp1, sp2))
    {
        image[displaced_pixel_coord] = float4(color, 1.0);
    }
}