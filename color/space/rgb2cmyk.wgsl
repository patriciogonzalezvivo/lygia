fn rgb2cmyk(rgb: vec3<f32>) -> vec4<f32> {
    let k = min(1.0 - rgb.r, min(1.0 - rgb.g, 1.0 - rgb.b));
    let invK = 1.0 - k;
    var cmy = (1.0 - rgb - k) / invK;
    cmy *= step(0.0, invK);
    return saturate(vec4<f32>(cmy, k));
}