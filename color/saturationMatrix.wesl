/*
contributors: Patricio Gonzalez Vivo
description: Generate a matrix to change the saturation of any color
use: saturationMatrix(<f32> amount)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn saturationMatrix(amount: f32) -> mat4x4f {
    let lum = vec3f(0.3086, 0.6094, 0.0820);
    let invAmount = 1.0 - amount;
    let r = vec3f(lum.x * invAmount) + vec3f(amount, 0.0, 0.0);
    let g = vec3f(lum.y * invAmount) + vec3f(0.0, amount, 0.0);
    let b = vec3f(lum.z * invAmount) + vec3f(0.0, 0.0, amount);
    return mat4x4f(
        vec4f(r, 0.0),
        vec4f(g, 0.0),
        vec4f(b, 0.0),
        vec4f(0.0, 0.0, 0.0, 1.0)
    );
}
