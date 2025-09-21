/*
contributors: Patricio Gonzalez Vivo
description: given a 3x3 returns a 4x4
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn toMat4(m: mat3x3<f32>) -> mat4x4<f32> {
    return mat4x4<f32>( vec4f(m[0], 0.0), 
                        vec4f(m[1], 0.0), 
                        vec4f(m[2], 0.0), 
                        vec4f(0.0, 0.0, 0.0, 1.0) );
}
