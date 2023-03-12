fn xyz2lab(c: vec3<f32>) -> vec3<f32> {
    let n = c / vec3<f32>(95.047, 100.0, 108.883);
    let c0 = pow(n, vec3<f32>(1.0 / 3.0));
    let c1 = (7.787 * n) + (16.0 / 116.0);
    let v = mix(c0, c1, step(n, vec3<f32>(0.008856)));
    return vec3<f32>(   (116.0 * v.y) - 16.0,
                        500.0 * (v.x - v.y),
                        200.0 * (v.y - v.z));
}