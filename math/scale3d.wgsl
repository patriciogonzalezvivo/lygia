/*
contributors: Patricio Gonzalez Vivo
description: returns a 3x3 scale matrix
*/

fn scale3d(s: vec3f) -> mat3x3<f32> {
    return mat3x3<f32>( s.x, 0.0, 0.0,
                        0.0, s.y, 0.0,
                        0.0, 0.0, s.z );
}
