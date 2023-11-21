fn exposure(color : vec3f, amount : f32) -> vec3f {
    return color * pow(2., amount);
}