fn rgb2YCbCr(rgb: vec3<f32>) -> vec3<f32> {
    let y = dot(rgb, vec3<f32>(0.299, 0.587, 0.114));
    let cb = 0.5 + dot(rgb, vec3<f32>(-0.168736, -0.331264, 0.5));
    let cr = 0.5 + dot(rgb, vec3<f32>(0.5, -0.418688, -0.081312));
    return vec3<f32>(y, cb, cr);
}
