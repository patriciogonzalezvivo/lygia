/*
contributors: Patricio Gonzalez Vivo
description: returns a 4x4 translate matrix
*/

fn translate4d(t: vec3f) -> mat4x4<f32> {
    return mat4x4<f32>( 1.0, 0.0, 0.0, 0.0,
                        0.0, 1.0, 0.0, 0.0,
                        0.0, 0.0, 1.0, 0.0,
                        t.x, t.y, t.z, 1.0 );
}
