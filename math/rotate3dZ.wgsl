/*
contributors: Patricio Gonzalez Vivo
description: returns a 3x3 rotation matrix
*/

fn rotate3dZ(r: f32) -> mat3x3<f32> {
    return mat3x3<f32>( vec3f(cos(r), -sin(r), 0.0),
                        vec3f(sin(r), cos(r), 0.0),
                        vec3f(0.0, 0.0, 1.0) );
}