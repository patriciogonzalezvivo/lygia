/*
contributors: Bjorn Ottosson (@bjornornorn)
description: oklab to linear RGB https://bottosson.github.io/posts/oklab/
use: <vec3\vec4> oklab2srgb(<vec3|vec4> oklab)
*/

#ifndef MAT_OKLAB2SRGB
#define MAT_OKLAB2SRGB
const mat3 OKLAB2SRGB_A = mat3(  
    1.0,            1.0,            1.0,
    0.3963377774,  -0.1055613458,  -0.0894841775,
    0.2158037573,  -0.0638541728,  -1.2914855480);
                       
const mat3 OKLAB2SRGB_B = mat3(  
    4.0767245293,  -1.2681437731,   -0.0041119885,
    -3.3072168827,  2.3098,          0.2307590544, 
    -0.3411344290,  1.7066093323231,-0.7034768625689);
#endif

#ifndef FNC_OKLAB2SRGB
#define FNC_OKLAB2SRGB
vec3 oklab2srgb(const in vec3 oklab) { 
    vec3 lms = OKLAB2SRGB_A * oklab;
    return OKLAB2SRGB_B * (lms * lms * lms);
}
vec4 oklab2srgb(const in vec4 oklab) { return vec4(oklab2srgb(oklab.xyz), oklab.a); }
#endif