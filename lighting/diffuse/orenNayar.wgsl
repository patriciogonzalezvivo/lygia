/*
contributors: Patricio Gonzalez Vivo
description: Calculate diffuse contribution using Oren and Nayar equation https://en.wikipedia.org/wiki/Oren%E2%80%93Nayar_reflectance_model
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn diffuseOrenNayar(L: vec3f, N: vec3f, V: vec3f, NoV: f32, NoL: f32, roughness: f32) -> f32{
    let LoV = dot(L, V);
    
    let s = LoV - NoL * NoV;
    let t = mix(1.0, max(NoL, NoV), step(0.0, s));

    let sigma2 = roughness * roughness;
    let A = 1.0 + sigma2 * (1.0 / (sigma2 + 0.13) + 0.5 / (sigma2 + 0.33));
    let B = 0.45 * sigma2 / (sigma2 + 0.09);

    return max(0.0, NoL) * (A + B * s / t);
}