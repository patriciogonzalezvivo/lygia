/*
contributors: Patricio Gonzalez Vivo
description: returns a 4x4 scale matrix
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn scale4d(s: vec3f) -> mat4x4<f32> {
    return mat4x4<f32>( s.x, 0.0, 0.0, 0.0,
                        0.0, s.y, 0.0, 0.0,
                        0.0, 0.0, s.z, 0.0,
                        0.0, 0.0, 0.0, 1.0 );
}
