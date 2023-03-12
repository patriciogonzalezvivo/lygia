// #ifndef SRGB_ALPHA
// #define SRGB_ALPHA 0.055
// #endif

fn srgb2rgb(srgb:vec3<f32>) -> vec3<f32> {
    let srgb_lo = srgb / 12.92;
    let srgb_hi = pow((srgb + 0.055)/(1.0 + 0.055), vec3<f32>(2.4));
    return mix(srgb_lo, srgb_hi, step(vec3<f32>(0.04045), srgb));
}