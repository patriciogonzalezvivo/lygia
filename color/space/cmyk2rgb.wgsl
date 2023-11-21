fn cmyk2rgb(cmyk: vec4f) -> vec3f {
    let invK: f32 = 1.0 - cmyk.w;
    return saturate(1.0 - min(vec3f(1.0), cmyk.xyz * invK + cmyk.w));
}
