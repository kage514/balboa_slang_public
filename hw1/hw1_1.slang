[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float2 center,
                  uniform float radius,
                  uniform float3 color,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    image[tid] = float4(0.5, 0.5, 0.5, 1.0);

    float2 pixel_center = float2(tid.xy) + 0.5;
    pixel_center.y = height - pixel_center.y;
    float2 offset = pixel_center - center;

    if(dot(offset, offset) < radius * radius)
        {
            image[tid] = float4(color, 1.0);
        }
}
