#include "../math/sum.hlsl"
#include "../sample/2DCube.hlsl"
#include "../sample.hlsl"

/*
original_author: Secret Weapons (@scrtwpns)
description: mix using mixbox pigment algorithm https://github.com/scrtwpns/pigment-mixing and converted to GLSL by Patricio Gonzalez Vivo
use: <float3\float4> mixBox(<float3|float4> rgbA, <float3|float4> rgbB, float pct)
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
#define MIXBOX_LATENT_TYPE float3x3
#endif

#ifndef FNC_MIXBOX
#define FNC_MIXBOX

float3 mixBox(float3 c) {
    float4 C = float4(c, 1.0 - sum(c));
    float4 S = C * C;
    float3 V = C.xxy * C.yzz;
    return  (C.x*S.x) * float3( 0.07717053,  0.02826978,  0.24832992) +
            (C.y*S.y) * float3( 0.95912302,  0.80256528,  0.03561839) +
            (C.z*S.z) * float3( 0.74683774,  0.04868586,  0.00000000) +
            (C.w*S.w) * float3( 0.99518138,  0.99978149,  0.99704802) +
            (S.x*C.y) * float3( 0.04819146,  0.83363781,  0.32515377) +
            (V.x*C.y) * float3(-0.68146950,  1.46107803,  1.06980936) +
            (S.x*C.z) * float3( 0.27058419, -0.15324870,  1.98735057) +
            (V.y*C.z) * float3( 0.80478189,  0.67093710,  0.18424500) +
            (S.x*C.w) * float3(-0.35031003,  1.37855826,  3.68865000) +
            (C.x*S.w) * float3( 1.05128046,  1.97815239,  2.82989073) +
            (S.y*C.z) * float3( 3.21607125,  0.81270228,  1.03384539) +
            (C.y*S.z) * float3( 2.78893374,  0.41565549, -0.04487295) +
            (S.y*C.w) * float3( 3.02162577,  2.55374103,  0.32766114) +
            (C.y*S.w) * float3( 2.95124691,  2.81201112,  1.17578442) +
            (S.z*C.w) * float3( 2.82677043,  0.79933038,  1.81715262) +
            (C.z*S.w) * float3( 2.99691099,  1.22593053,  1.80653661) +
            (V.x*C.z) * float3( 1.87394106,  2.05027182, -0.29835996) +
            (V.x*C.w) * float3( 2.56609566,  7.03428198,  0.62575374) +
            (V.y*C.w) * float3( 4.08329484, -1.40408358,  2.14995522) +
            (V.z*C.w) * float3( 6.00078678,  2.55552042,  1.90739502);
}

MIXBOX_LATENT_TYPE mixBox_rgb2latent(float3 rgb) {
    rgb = saturate(rgb);
    float3 lut = sample2DCube(MIXBOX_LUT, rgb).xyz;
    return MIXBOX_LATENT_TYPE(lut, rgb - mixBox(lut), float3(0.0, 0.0, 0.0));
}

float3 mixBox_latent2rgb(MIXBOX_LATENT_TYPE latent) {
    return saturate( mixBox(latent[0]) + latent[1] );
}

float3 mixBox(float3 colA, float3 colB, float t) {
    return mixBox_latent2rgb((1.0-t) * mixBox_rgb2latent(colA) + t * mixBox_rgb2latent(colB));
}

float4 mixBox(float4 colA, float4 colB, float t) {
    return float4(mixBox(colA.rgb, colB.rgb, t), lerp(colA.a, colB.a, t));
}

float3 mixBox(float4 colA, float4 colB, float4 colC) {
    MIXBOX_LATENT_TYPE cA = mixBox_rgb2latent(colA.rgb);
    MIXBOX_LATENT_TYPE cB = mixBox_rgb2latent(colB.rgb);
    MIXBOX_LATENT_TYPE cC = mixBox_rgb2latent(colC.rgb);
    return mixBox_latent2rgb( cA * colA.a + cB*colB.a + cC*colC.a );
}
#endif
