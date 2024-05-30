/*
contributors: Patricio Gonzalez Vivo
description: returns a 3x3 rotation matrix
*/

fn rotate3dY(r: f32) -> mat3x3<f32> {
    return mat3x3<f32>(   vec3f(cos(r), 0.0, -sin(r)),
                        vec3f(0.0, 1.0, 0.0),
                        vec3f(sin(r), 0.0, cos(r)) );
}