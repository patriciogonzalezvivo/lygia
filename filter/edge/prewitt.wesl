/*
contributors: Brad Larson
description: Adapted version of Prewitt edge detection from https://github.com/BradLarson/GPUImage2
examples:
    - /shaders/filter_edge2D.frag

*/

fn edgePrewitt(tex: texture_2d<f32>, samp: sampler, uv: vec2f, offset: vec2f) -> vec3f {
    let top_left = textureSampleBaseClampToEdge(tex, samp, uv + vec2f(-offset.x, offset.y)).xyz;
    let left = textureSampleBaseClampToEdge(tex, samp, uv + vec2f(-offset.x, 0.0)).xyz;
    let bottom_left = textureSampleBaseClampToEdge(tex, samp, uv + vec2f(-offset.x, -offset.y)).xyz;
    let top = textureSampleBaseClampToEdge(tex, samp, uv + vec2f(0.0, offset.y)).xyz;
    let bottom = textureSampleBaseClampToEdge(tex, samp, uv + vec2f(0.0, -offset.y)).xyz;
    let top_right = textureSampleBaseClampToEdge(tex, samp, uv + offset).xyz;
    let right = textureSampleBaseClampToEdge(tex, samp, uv + vec2f(offset.x, 0.0)).xyz;
    let bottom_right = textureSampleBaseClampToEdge(tex, samp, uv + vec2f(offset.x, -offset.y)).xyz;
    let x = -top_left - top - top_right + bottom_left + bottom + bottom_right;
    let y = -bottom_left - left - top_left + bottom_right + right + top_right;
    return sqrt((x * x) + (y * y));
}
