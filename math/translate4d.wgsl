/*
contributors: Patricio Gonzalez Vivo
description: returns a 4x4 translate matrix
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn translate4d(t: vec3f) -> mat4x4<f32> {
    return mat4x4<f32>( 1.0, 0.0, 0.0, 0.0,
                        0.0, 1.0, 0.0, 0.0,
                        0.0, 0.0, 1.0, 0.0,
                        t.x, t.y, t.z, 1.0 );
}
