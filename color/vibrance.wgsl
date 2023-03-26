#include "luma.wgsl"

fn vibrance(color : vec3<f32>, v:f32) -> vec3<f32> {
    let max_color = max(color.r, max(color.g, color.b));
    let min_color = min(color.r, min(color.g, color.b));
    let sat = max_color - min_color;
    return mix(vec3<f32>( luma(color) ), color, 1.0 + (v * 1.0 - (sign(v) * sat)));
}