fn linear2gamma(v : vec3<f32>) -> vec3<f32> {
    return pow(v, vec3<f32>(1. / 2.2));
}
