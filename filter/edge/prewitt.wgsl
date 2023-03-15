fn edgePrewitt(tex: texture_2d<f32>, samp: sampler, uv: vec2<f32>, offset: vec2<f32>) -> vec3<f32> {
    let top_left = textureSample(tex, samp, uv + vec2<f32>(-offset.x, offset.y)).xyz;
    let left = textureSample(tex, samp, uv + vec2<f32>(-offset.x, 0.)).xyz;
    let bottom_left = textureSample(tex, samp, uv + vec2<f32>(-offset.x, -offset.y)).xyz;
    let top = textureSample(tex, samp, uv + vec2<f32>(0., offset.y)).xyz;
    let bottom = textureSample(tex, samp, uv + vec2<f32>(0., -offset.y)).xyz;
    let top_right = textureSample(tex, samp, uv + offset).xyz;
    let right = textureSample(tex, samp, uv + vec2<f32>(offset.x, 0.)).xyz;
    let bottom_right = textureSample(tex, samp, uv + vec2<f32>(offset.x, -offset.y)).xyz;
    let x = -top_left - top - top_right + bottom_left + bottom + bottom_right;
    let y = -bottom_left - left - top_left + bottom_right + right + top_right;
    return sqrt((x * x) + (y * y));
}