import hw3_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  uniform Camera camera,
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Sphere> spheres,
                  StructuredBuffer<PointLight> lights,
                  uniform int num_lights,
                  uniform int max_depth,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    float2 uv = (float2(tid) + 0.5) / float2(width, height);
    float aspect = float(width) / float(height);

    float plane_half_width = tan(radians(camera.fov_x) / 2.0);
    float plane_half_height = plane_half_width / aspect;
    
    float3 camera_dir = float3(
        (2.0 * uv.x - 1.0) * plane_half_width,
        (1.0 - 2.0 * uv.y) * plane_half_height,
        -1.0
    );

    float3 ray_origin = xform_pt(camera.cam_to_world, float3(0.0));
    float3 ray_dir = xform_vec(camera.cam_to_world, camera_dir);

    float3 color = float3(0.0);
    float3 throughput = float3(1.0);

    for(int bounce = 0; bounce < max_depth; bounce++)
    {
        float closest_t = 0.0;
        int closest_shape = -1;

        //calculating the closest hitten shape.
        for (int i = 0; i < num_shapes; i++) 
        {
            Sphere sphere = spheres[shapes[i].index];
            
            float3 local_origin = xform_pt(shapes[i].world_to_obj, ray_origin);
            float3 local_dir = xform_vec(shapes[i].world_to_obj, ray_dir);

            float3 offset = local_origin - sphere.center;

            float a = dot(local_dir, local_dir);
            float b = 2.0 * dot(offset, local_dir);
            float c = dot(offset, offset) - sphere.radius * sphere.radius;

            float2 roots;
            if (!solve_quadratic(a, b, c, roots)) 
            {
                continue;
            }

            for (int j = 0; j < 2; ++j) 
            {
                float t = roots[j];
                if (t > EPSILON && (closest_shape < 0 || t < closest_t)) 
                {
                    closest_t = t;
                    closest_shape = i;
                }
            }
        }

        //no more bounces
        if(closest_shape < 0)
        {
            color += throughput * background;
            break;
        }

        float3 hit_point = ray_origin + closest_t * ray_dir;
        Shape hit_shape = shapes[closest_shape];

        float3 local_hit = xform_pt(hit_shape.world_to_obj, hit_point);
        float3 local_normal = local_hit - spheres[hit_shape.index].center;
        float3 world_normal = normalize(
            xform_vec(transpose(hit_shape.world_to_obj), local_normal)
        );

        //Get the diffuse light and shadow effect
        float3 diffuse_light = float3(0.0);
        if (closest_shape >= 0) 
        {
            for(int i = 0; i < num_lights; i++)
            {
                PointLight light = lights[i];
                float3 shadow_ray = light.position - hit_point;
                float shadow_a = dot(shadow_ray, shadow_ray);
                bool blocked = false;

                for (int i = 0; i < num_shapes; i++)
                {
                    Sphere sphere = spheres[shapes[i].index];

                    float3 local_origin = xform_pt(shapes[i].world_to_obj, hit_point);
                    float3 local_dir = xform_vec(shapes[i].world_to_obj, shadow_ray);
                    float3 offset = local_origin - sphere.center;

                    float local_shadow_a = dot(local_dir, local_dir);
                    float local_shadow_b = 2.0 * dot(offset, local_dir);
                    float local_shadow_c = dot(offset, offset) - sphere.radius * sphere.radius;

                    float2 roots;
                    if (!solve_quadratic(local_shadow_a, local_shadow_b, local_shadow_c, roots)) 
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
                    float cos_theta = max(dot(world_normal, normalize(shadow_ray)), 0.0);
                    diffuse_light += hit_shape.Kd * light.intensity * cos_theta / shadow_a;
                }
            }
        }

        color += throughput * diffuse_light;

        //Calculating the throughput for the bounce and reflect the ray
        throughput *= hit_shape.Kr;
        if (!any(throughput > 0.0))
        {
            break;
        }

        ray_origin = hit_point;
        ray_dir = reflect(ray_dir, world_normal);
    }

    image[tid] = float4(color, 1.0);
}
