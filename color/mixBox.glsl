#include "../math/saturate.glsl"
#include "../math/sum.glsl"
#include "../sample/2DCube.glsl"
#include "../sample.glsl"

/*
original_author: Secret Weapons (@scrtwpns)
description: mix using mixbox pigment algorithm https://github.com/scrtwpns/pigment-mixing 
use: <vec3\vec4> mixBox(<vec3|vec4> rgbA, <vec3|vec4> rgbB, float pct)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - MIXBOX_LUT: name of the texture uniform which you can find here https://github.com/scrtwpns/pigment-mixing
    - MIXBOX_LUT_FLIP_Y: when defined it expects a vertically flipled texture  
    - MIXBOX_LUT_SAMPLER_FNC: sampler function. Default, texture2D(MIXBOX_LUT, POS_UV).rgb
    - MIXBOX_LUT_CELL_SIZE: Default 256
license: |
    Copyright (c) 2022, Secret Weapons. All rights reserved.
    This code is for non-commercial use only. It is provided for research and evaluation purposes.
    If you wish to obtain commercial license, please contact: mixbox@scrtwpns.com
*/

#ifndef MIXBOX_LUT 
#define MIXBOX_LUT u_tex0
#endif

#ifndef MIXBOX_LUT_SAMPLER_FNC
#define MIXBOX_LUT_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV).rgb
#endif

#ifndef MIXBOX_LATENT_TYPE
#define MIXBOX_LATENT_TYPE mat3
#endif

#ifndef FNC_MIXBOX
#define FNC_MIXBOX

vec3 mixBox(vec3 c) {
    vec4 C = vec4(c, 1.0 - sum(c));
    vec4 S = C * C;
    vec3 V = C.xxy * C.yzz;
    return  (C.x*S.x) * vec3( 0.07717053,  0.02826978,  0.24832992) +
            (C.y*S.y) * vec3( 0.95912302,  0.80256528,  0.03561839) +
            (C.z*S.z) * vec3( 0.74683774,  0.04868586,  0.00000000) +
            (C.w*S.w) * vec3( 0.99518138,  0.99978149,  0.99704802) +
            (S.x*C.y) * vec3( 0.04819146,  0.83363781,  0.32515377) +
            (V.x*C.y) * vec3(-0.68146950,  1.46107803,  1.06980936) +
            (S.x*C.z) * vec3( 0.27058419, -0.15324870,  1.98735057) +
            (V.y*C.z) * vec3( 0.80478189,  0.67093710,  0.18424500) +
            (S.x*C.w) * vec3(-0.35031003,  1.37855826,  3.68865000) +
            (C.x*S.w) * vec3( 1.05128046,  1.97815239,  2.82989073) +
            (S.y*C.z) * vec3( 3.21607125,  0.81270228,  1.03384539) +
            (C.y*S.z) * vec3( 2.78893374,  0.41565549, -0.04487295) +
            (S.y*C.w) * vec3( 3.02162577,  2.55374103,  0.32766114) +
            (C.y*S.w) * vec3( 2.95124691,  2.81201112,  1.17578442) +
            (S.z*C.w) * vec3( 2.82677043,  0.79933038,  1.81715262) +
            (C.z*S.w) * vec3( 2.99691099,  1.22593053,  1.80653661) +
            (V.x*C.z) * vec3( 1.87394106,  2.05027182, -0.29835996) +
            (V.x*C.w) * vec3( 2.56609566,  7.03428198,  0.62575374) +
            (V.y*C.w) * vec3( 4.08329484, -1.40408358,  2.14995522) +
            (V.z*C.w) * vec3( 6.00078678,  2.55552042,  1.90739502);
}

MIXBOX_LATENT_TYPE mixBox_rgb2latent(vec3 rgb) {
    rgb = saturate(rgb);
    vec3 lut = sample2DCube(MIXBOX_LUT, rgb).xyz;
    return MIXBOX_LATENT_TYPE(lut, rgb - mixBox(lut), vec3(0.0, 0.0, 0.0));
}

vec3 mixBox_latent2rgb(MIXBOX_LATENT_TYPE latent) {
    return saturate( mixBox(latent[0]) + latent[1] );
}

vec3 mixBox(vec3 colA, vec3 colB, float t) {
    return mixBox_latent2rgb((1.0-t) * mixBox_rgb2latent(colA) + t * mixBox_rgb2latent(colB));
}

vec4 mixBox(vec4 colA, vec4 colB, float t) {
    return vec4(mixBox(colA.rgb, colB.rgb, t), mix(colA.a, colB.a, t));
}

vec3 mixBox(vec4 colA, vec4 colB, vec4 colC) {
    MIXBOX_LATENT_TYPE cA = mixBox_rgb2latent(colA.rgb);
    MIXBOX_LATENT_TYPE cB = mixBox_rgb2latent(colB.rgb);
    MIXBOX_LATENT_TYPE cC = mixBox_rgb2latent(colC.rgb);
    return mixBox_latent2rgb( cA * colA.a + cB*colB.a + cC*colC.a );
}
#endif
