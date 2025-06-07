/*
contributors: Bjorn Ottosson (@bjornornorn)
description: Oklab to linear RGB https://bottosson.github.io/posts/oklab/
license: 
    - MIT License (MIT) Copyright (c) 2020 Björn Ottosson
*/

const OKLAB2RGB_A : mat3x3<f32>  = mat3x3<f32>(
    vec3f(1.0, 1.0, 1.0),
    vec3f(0.3963377774, -0.1055613458, -0.0894841775),
    vec3f(0.2158037573, -0.0638541728, -1.2914855480) );

const OKLAB2RGB_B : mat3x3<f32>  = mat3x3<f32>(
    vec3f(4.0767416621, -1.2684380046, -0.0041960863),
    vec3f(-3.3077115913, 2.6097574011, -0.7034186147),
    vec3f(0.2309699292, -0.3413193965, 1.7076147010) );

fn oklab2rgb(oklab: vec3f) -> vec3f {
    let lms = OKLAB2RGB_A * oklab;
    return OKLAB2RGB_B * (lms * lms * lms);
}