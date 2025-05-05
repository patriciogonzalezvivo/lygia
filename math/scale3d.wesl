/*
contributors: Patricio Gonzalez Vivo
description: returns a 3x3 scale matrix
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn scale3d(s: vec3f) -> mat3x3<f32> {
    return mat3x3<f32>( s.x, 0.0, 0.0,
                        0.0, s.y, 0.0,
                        0.0, 0.0, s.z );
}
