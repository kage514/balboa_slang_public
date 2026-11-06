import hw3_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  StructuredBuffer<Sphere> spheres,
                  uniform int num_spheres,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    float2 uv = (float2(tid) + 0.5) / float2(width, height);
    float aspect = float(width) / float(height);
    float3 ray_dir = float3((2.0 * uv.x - 1.0) * aspect, 1.0 - 2.0 * uv.y, -1.0);

    float a = dot(ray_dir, ray_dir);
    float closest_t = 0.0;
    int closest_sphere = -1;

    for (int i = 0; i < num_spheres; i++) 
    {
        Sphere sphere = spheres[i];
        float3 offset = -sphere.center;
        float b = 2.0 * dot(offset, ray_dir);
        float c = dot(offset, offset) - sphere.radius * sphere.radius;

        float2 roots;
        if (!solve_quadratic(a, b, c, roots)) 
        {
            continue;
        }

        for (int j = 0; j < 2; ++j) 
        {
            float t = roots[j];
            if (t > 0.0 && (closest_sphere < 0 || t < closest_t)) 
            {
                closest_t = t;
                closest_sphere = i;
            }
        }
    }

    float3 color = background;
    if (closest_sphere >= 0) 
    {
        float3 hit_point = closest_t * ray_dir;
        float3 normal = normalize(hit_point - spheres[closest_sphere].center);
        color = (normal + 1.0) / 2.0;
    }
    image[tid] = float4(color, 1.0);
}
