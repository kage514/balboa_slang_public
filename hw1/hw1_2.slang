import hw1_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  StructuredBuffer<float2> polyline,
                  uniform int num_polyline,
                  uniform bool is_closed,
                  uniform bool use_fill_color,
                  uniform float3 fill_color,
                  uniform bool use_stroke_color,
                  uniform float3 stroke_color,
                  uniform float stroke_width,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    image[tid] = float4(0.5, 0.5, 0.5, 1.0);

    float2 pixel_center = float2(tid.xy) + 0.5;
    pixel_center.y = height - pixel_center.y;

    //rendering filling
    if(is_closed && use_fill_color)
    {
        int intersection_count = 0;
        for(int i = 0; i < num_polyline; i++)
        {
            int next_i = (i + 1) % num_polyline;
            float t = (pixel_center.y - polyline[i].y) / (polyline[next_i].y - polyline[i].y);
            float s = polyline[i].x + t * (polyline[next_i].x - polyline[i].x) - tid.x;
            if(s >= 0.0 && t >= 0.0 && t <= 1.0)
            {
                if(polyline[next_i].y - polyline[i].y >= 0.0)
                {
                    intersection_count++;
                }
                else
                {
                    intersection_count--;
                }
            }
        }
        if(intersection_count != 0)
        {
            image[tid] = float4(fill_color, 1.0);
        }
    }

    //rendering strokes
    if(use_stroke_color)
    {
        int num_segments = is_closed ? num_polyline : num_polyline - 1;
        for(int i = 0; i < num_segments; i++)
        {
            int next_i = (i + 1) % num_polyline;
            float2 normalized_dir = normalize(polyline[next_i] - polyline[i]);
            float l = dot(normalized_dir, (pixel_center - polyline[i]));
            float2 projected_pixel_center = polyline[i] + l * normalized_dir;
            if(l > 0 && l < length(polyline[next_i] - polyline[i]) && length(pixel_center - projected_pixel_center) < stroke_width / 2)
            {
                image[tid] = float4(stroke_color, 1.0);
            }
        }
    }

    //draw round joins
    float radius = stroke_width / 2;
    if(is_closed)
    {
        for(int i = 0; i < num_polyline; i++)
        {
            if(length(pixel_center - polyline[i]) < radius)
            {
                image[tid] = float4(stroke_color, 1.0);
            }
        }
    }
    else
    {
        for(int i = 1; i < num_polyline - 1; i++)
        {
            if(length(pixel_center - polyline[i]) < radius)
            {
                image[tid] = float4(stroke_color, 1.0);
            }
        }
    }
}

