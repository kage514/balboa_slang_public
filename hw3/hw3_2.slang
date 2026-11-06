import hw3_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Sphere> spheres,
                  StructuredBuffer<PointLight> lights,
                  uniform int num_lights,
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
    int closest_shape = -1;

    for (int i = 0; i < num_shapes; i++) 
    {
        Sphere sphere = spheres[shapes[i].index];
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
            if (t > 0.0 && (closest_shape < 0 || t < closest_t)) 
            {
                closest_t = t;
                closest_shape = i;
            }
        }
    }

    float3 color = closest_shape >= 0 ? float3(0.0) : background;
    if (closest_shape >= 0) 
    {
        float3 hit_point = closest_t * ray_dir;
        Shape hit_shape = shapes[closest_shape];
        float3 normal = normalize(hit_point - spheres[hit_shape.index].center);
        for(int i = 0; i < num_lights; i++)
        {
            PointLight light = lights[i];
            float3 shadow_ray = light.position - hit_point;
            float shadow_a = dot(shadow_ray, shadow_ray);
            bool blocked = false;

            for (int i = 0; i < num_shapes; i++)
            {
                Sphere sphere = spheres[shapes[i].index];
                float3 offset = hit_point - sphere.center;
                float shadow_b = 2.0 * dot(offset, shadow_ray);
                float shadow_c = dot(offset, offset) - sphere.radius * sphere.radius;

                float2 roots;
                if (!solve_quadratic(shadow_a, shadow_b, shadow_c, roots)) 
                {
                    continue;
                }

                for (int j = 0; j < 2; ++j) 
                {
                    float t = roots[j];
                    if (t > EPSILON && t < 1.0) 
                    {
                        blocked = true;
                        break;
                    }
                }
                if(blocked)
                {
                    break;
                }
            }

            if(!blocked)
            {
                float cos_theta = max(dot(normal, normalize(shadow_ray)), 0.0);
                color += hit_shape.Kd * light.intensity * cos_theta / shadow_a;
            }
        }
    }
    image[tid] = float4(color, 1.0);
}