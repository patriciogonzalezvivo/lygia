/*
contributors: Johan Ismael
description: Similar to step but for an interval instead of a threshold. Returns 1 is x is between left and right, 0 otherwise
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn within(x: f32, _min: f32, _max: f32) -> f32 {
    return step(_min, x) * (1. - step(_max, x));
}

fn within2(x: vec2f, _min: vec2f, _max: vec2f) -> f32 {
    let rta = step(_min, x) * (1. - step(_max, x));
    return rta.x * rta.y;
}

fn within3(x: vec3f, _min: vec3f, _max: vec3f) -> f32 {
    let rta = step(_min, x) * (1. - step(_max, x));
    return rta.x * rta.y * rta.z;
}

fn within4(x: vec4f, _min: vec4f, _max: vec4f) -> f32 {
    let rta = step(_min, x) * (1. - step(_max, x));
    return rta.x * rta.y * rta.z * rta.w;
}
