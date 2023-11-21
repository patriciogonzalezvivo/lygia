fn rgb2YCbCr(rgb: vec3f) -> vec3f {
    let y = dot(rgb, vec3f(0.299, 0.587, 0.114));
    let cb = 0.5 + dot(rgb, vec3f(-0.168736, -0.331264, 0.5));
    let cr = 0.5 + dot(rgb, vec3f(0.5, -0.418688, -0.081312));
    return vec3f(y, cb, cr);
}
