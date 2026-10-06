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

    if(is_closed && use_fill_color)
    {
        for(int i = 0; i < num_polyline; i++)
        {
            int next_i = (i + 1) % num_polyline;
            float t = (pixel_center.y - polyline[i].y) / (polyline[i + 1].y - polyline[i].y);
            float s = polyline[i].x + t * (polyline[i + 1].x - polyline[i].x) - tid.x;
            if(s >= 0 && t >= 0 && t <= 1.0)
            {
                image[tid] = float4(fill_color, 1.0);
            }
        }
    }
}
