/*
contributors: Patricio Gonzalez Vivo
description: returns a 4x4 scale matrix
*/

fn scale4d(s: vec3f) -> mat4x4<f32> {
    return mat4x4<f32>( s.x, 0.0, 0.0, 0.0,
                        0.0, s.y, 0.0, 0.0,
                        0.0, 0.0, s.z, 0.0,
                        0.0, 0.0, 0.0, 1.0);
}
