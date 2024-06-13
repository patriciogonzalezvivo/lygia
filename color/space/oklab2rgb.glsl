/*
contributors: Bjorn Ottosson (@bjornornorn)
description: Oklab to linear RGB https://bottosson.github.io/posts/oklab/
use: <vec3\vec4> oklab2rgb(<vec3|vec4> oklab)
license: 
    - MIT License (MIT) Copyright (c) 2020 Björn Ottosson
*/

#ifndef MAT_OKLAB2RGB
#define MAT_OKLAB2RGB
const mat3 OKLAB2RGB_A = mat3(
    1.0,           1.0,           1.0,
    0.3963377774, -0.1055613458, -0.0894841775,
    0.2158037573, -0.0638541728, -1.2914855480);

const mat3 OKLAB2RGB_B = mat3(
    4.0767416621, -1.2684380046, -0.0041960863,
    -3.3077115913, 2.6097574011, -0.7034186147,
    0.2309699292, -0.3413193965, 1.7076147010);
#endif

#ifndef FNC_OKLAB2RGB
#define FNC_OKLAB2RGB
vec3 oklab2rgb(const in vec3 oklab) {
    vec3 lms = OKLAB2RGB_A * oklab;
    return OKLAB2RGB_B * (lms * lms * lms);
}
vec4 oklab2rgb(const in vec4 oklab) { return vec4(oklab2rgb(oklab.xyz), oklab.a); }
#endif
