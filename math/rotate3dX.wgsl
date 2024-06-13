/*
contributors: Patricio Gonzalez Vivo
description: returns a 3x3 rotation matrix
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn rotate3dX(r: f32) -> mat3x3<f32> {
    return mat3x3<f32>( vec3f(1.0, 0.0, 0.0),
                        vec3f(0.0, cos(r), -sin(r)),
                        vec3f(0.0, sin(r), cos(r)) );
}