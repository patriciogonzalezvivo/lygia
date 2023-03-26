fn exposure(color : vec3<f32>, amount : f32) -> vec3<f32> {
    return color * pow(2., amount);
}