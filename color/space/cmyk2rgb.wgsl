fn cmyk2rgb(cmyk: vec4<f32>) -> vec3<f32> {
    let invK: f32 = 1.0 - cmyk.w;
    return saturate(1.0 - min(vec3<f32>(1.0), cmyk.xyz * invK + cmyk.w));
}
