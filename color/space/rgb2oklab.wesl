/*
contributors: Bjorn Ottosson (@bjornornorn)
description: |
    Linear rgb to OKLab https://bottosson.github.io/posts/oklab/
use: <vec3\vec4> rgb2oklab(<vec3|vec4> rgb)
license:
    - MIT License (MIT) Copyright (c) 2020 Björn Ottosson
*/

const RGB2OKLAB_A : mat3x3<f32>  = mat3x3<f32>( vec3f(0.2104542553, 1.9779984951, 0.0259040371),
                                                vec3f(0.7936177850, -2.4285922050, 0.7827717662),
                                                vec3f(-0.0040720468, 0.4505937099, -0.8086757660) );

const RGB2OKLAB_B : mat3x3<f32>  = mat3x3<f32>( vec3f(0.4122214708, 0.2119034982, 0.0883024619),
                                                vec3f(0.5363325363, 0.6806995451, 0.2817188376),
                                                vec3f(0.0514459929, 0.1073969566, 0.6299787005) );

fn rgb2oklab(rgb: vec3f) -> vec3f {
    let lms = RGB2OKLAB_B * rgb;
    return RGB2OKLAB_A * (sign(lms) * pow(abs(lms), vec3f(0.3333333333333)));
}