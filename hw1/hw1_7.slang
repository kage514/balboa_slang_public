import hw1_common;

[shader("compute")]
[numthreads(8, 8, 1)]
void compute_main(uint2 tid : SV_DispatchThreadID,
                  uniform float3 background,
                  uniform int samples_per_axis,
                  StructuredBuffer<Shape> shapes,
                  uniform int num_shapes,
                  StructuredBuffer<Circle> circles,
                  StructuredBuffer<Polyline> polylines,
                  StructuredBuffer<QuadraticBezier> /*quadratic_beziers*/, // ignore for now
                  StructuredBuffer<float2> points,
                  RWTexture2D<float4> image) {
    uint width, height;
    image.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height) {
        return;
    }

    image[tid] = float4(background, 1.0);

    float2 pixel_origin = float2(tid.xy);
    float4 avg_color = float4(0.0);

    for(int xx = 0; xx < samples_per_axis; xx++)
    {
        for(int yy = 0; yy < samples_per_axis; yy++)
        {
            float4 sample_color = float4(background, 1.0);

            float2 sample_pos = pixel_origin + float2(
            (xx + 0.5) / float(samples_per_axis),
            (yy + 0.5) / float(samples_per_axis));
            sample_pos.y = height - sample_pos.y;
            
            for(int i = num_shapes - 1; i >= 0; i--)
            {
                Shape cur_shape = shapes[i];
                float3 obj_pos_f3 = mul(cur_shape.world_to_obj, float3(sample_pos, 1.0));
                float2 obj_pos = obj_pos_f3.xy;
                if(cur_shape.type == ShapeType.Circle)
                {
                    float2 offset = circles[cur_shape.index].center - obj_pos;
                    if(cur_shape.use_fill)
                    {
                        if(dot(offset, offset) < circles[cur_shape.index].radius * circles[cur_shape.index].radius)
                        {
                            float alpha = cur_shape.fill_rgba.a;

                            sample_color.rgb = alpha * cur_shape.fill_rgba.rgb
                                + (1.0 - alpha) * sample_color.rgb;
                        }
                    }
                    if(cur_shape.use_stroke)
                    {
                        if(dot(offset, offset) <= (circles[cur_shape.index].radius + cur_shape.stroke_width / 2) * (circles[cur_shape.index].radius + cur_shape.stroke_width / 2)
                            && dot(offset, offset) >= (circles[cur_shape.index].radius - cur_shape.stroke_width / 2) * (circles[cur_shape.index].radius - cur_shape.stroke_width / 2))
                        {
                            float alpha = cur_shape.stroke_rgba.a;

                            sample_color.rgb = alpha * cur_shape.stroke_rgba.rgb
                                + (1.0 - alpha) * sample_color.rgb;
                        }
                    }
                }
                else if(cur_shape.type == ShapeType.Polyline)
                {
                    //rendering filling
                    Polyline cur_polyline = polylines[cur_shape.index];
                    if(cur_polyline.closed && cur_shape.use_fill)
                    {
                        int intersection_count = 0;
                        for(int i = cur_polyline.pts_begin; i < cur_polyline.pts_end; i++)
                        {
                            int next_i = (i + 1);
                            if(next_i >= cur_polyline.pts_end)
                            {
                                next_i = cur_polyline.pts_begin;
                            }
                            float t = (obj_pos.y - points[i].y) / (points[next_i].y - points[i].y);
                            float s = points[i].x + t * (points[next_i].x - points[i].x) - obj_pos.x;
                            if(s >= 0.0 && t >= 0.0 && t <= 1.0)
                            {
                                if(points[next_i].y - points[i].y >= 0.0)
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
                            float alpha = cur_shape.fill_rgba.a;

                            sample_color.rgb = alpha * cur_shape.fill_rgba.rgb
                                + (1.0 - alpha) * sample_color.rgb;
                        }
                    }

                    //rendering strokes
                    bool stroke_hit = false; //for avoiding rendering both the stroke and round join, causing alpha problem.
                    if(cur_shape.use_stroke)
                    {
                        int correct_pts_end = cur_polyline.closed ? cur_polyline.pts_end : cur_polyline.pts_end - 1;
                        for(int i = cur_polyline.pts_begin; i < correct_pts_end; i++)
                        {
                            int next_i = (i + 1);
                            if(next_i >= cur_polyline.pts_end)
                            {   
                                next_i = cur_polyline.pts_begin;
                            }
                            float2 normalized_dir = normalize(points[next_i] - points[i]);
                            float l = dot(normalized_dir, (obj_pos - points[i]));
                            float2 projected_pixel_center = points[i] + l * normalized_dir;
                            if(l > 0 && l < length(points[next_i] - points[i]) && length(obj_pos - projected_pixel_center) <= cur_shape.stroke_width / 2)
                            {
                                stroke_hit = true;
                            }
                        }

                        //draw round joins
                        float radius = cur_shape.stroke_width / 2;
                        if(cur_polyline.closed)
                        {
                            for(int i = cur_polyline.pts_begin; i < cur_polyline.pts_end; i++)
                            {
                                if(length(obj_pos - points[i]) < radius)
                                {
                                    stroke_hit = true;
                                }
                            }
                        }
                        else
                        {
                            for(int i = cur_polyline.pts_begin + 1; i < cur_polyline.pts_end - 1; i++)
                            {
                                if(length(obj_pos - points[i]) < radius)
                                {
                                    stroke_hit = true;
                                }
                            }
                        }

                        if(stroke_hit)
                        {
                            float alpha = cur_shape.stroke_rgba.a;

                            sample_color.rgb = alpha * cur_shape.stroke_rgba.rgb
                                + (1.0 - alpha) * sample_color.rgb;
                        }
                    }
                }
            }
            avg_color += sample_color / float(samples_per_axis * samples_per_axis);
        }
    }

    image[tid] = avg_color;
}
