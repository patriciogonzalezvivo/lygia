fn hsv2rgb(hsb : vec3<f32>) -> vec3<f32> {
    var rgb = saturate(abs(mod(hsb.x * 6.0 + vec3<f32>(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0);
    // #ifdef HSV2RGB_SMOOTH
    // rgb = rgb*rgb*(3. - 2. * rgb);
    // #endif
    return hsb.z * mix(vec3(1.), rgb, hsb.y);
}
