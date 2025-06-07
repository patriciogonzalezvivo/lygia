fn schlick(f0: vec3f, f90: vec3f, cos0: f32) -> vec3f {
    return f0 + (f90 - f0) * pow(1.0 - cos0, 5.0);
}