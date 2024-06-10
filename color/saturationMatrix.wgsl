/*
contributors: Patricio Gonzalez Vivo
description: Generate a matrix to change a the saturation of any color
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn saturationMatrix(amount : f32) -> mat3x3<f32> {
    let lum = vec3f(0.3086, 0.6094, 0.0820 );
    let invAmount = 1.0 - amount;
    return mat3x3<f32>( vec3f(lum.x * invAmount) + vec3f(amount, .0, .0), 
                        vec3f(lum.y * invAmount) + vec3f( .0, amount, .0),
                        vec3f(lum.z * invAmount) + vec3f( .0, .0, amount));
}