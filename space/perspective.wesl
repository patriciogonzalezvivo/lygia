/*
contributors: Patricio Gonzalez Vivo
description: create a perspective matrix
use: <mat4> perspective(<float> fov, <float> aspect, <float> near, <float> far);
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn perspective(fov: f32, aspect: f32, near: f32, far: f32) -> mat4x4f {
    let f = 1.0 / tan(fov / 2.0);
    let nf = 1.0 / (near - far);
    return mat4x4f(
        f / aspect, 0.0, 0.0, 0.0,
        0.0, f, 0.0, 0.0,
        0.0, 0.0, (far + near) * nf, -1.0,
        0.0, 0.0, (2.0 * far * near) * nf, 0.0
    );
}
