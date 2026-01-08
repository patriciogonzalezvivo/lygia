import lygia::math::pow5::pow5;

// Schlick 1994, "An Inexpensive BRDF Model for Physically-Based Rendering"
fn schlick(f0: vec3f, f90: f32, VoH: f32) -> vec3f {
    let f = pow5(1.0 - VoH);
    return f + f0 * (f90 - f);
}

fn schlickVec3(f0: vec3f, f90: vec3f, VoH: f32) -> vec3f {
    return f0 + (f90 - f0) * pow5(1.0 - VoH);
}

fn schlickF32(f0: f32, f90: f32, VoH: f32) -> f32 {
    return f0 + (f90 - f0) * pow5(1.0 - VoH);
}
