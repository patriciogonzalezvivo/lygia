/*
contributors: Patricio Gonzalez Vivo
description: gaussian coefficient
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn gaussian(d: f32, sigma: f32) -> f32 { return exp(-(d*d) / (2.0 * sigma * sigma)); }
fn gaussian2(d: vec2f, sigma: f32) -> f32 { return exp(-(d.x*d.x + d.y*d.y) / (2.0 * sigma * sigma));  }
fn gaussian3(d: vec3f, sigma: f32) -> f32 { return exp(-(d.x*d.x + d.y*d.y + d.z*d.z) / (2.0 * sigma * sigma)); }
fn gaussian4(d: vec4f, sigma: f32) -> f32 { return exp(-(d.x*d.x + d.y*d.y + d.z*d.z + d.w*d.w) / (2.0 * sigma * sigma)); }
