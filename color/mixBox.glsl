#include "../math/saturate.glsl"

/*
original_author: Secret Weapons (@scrtwpns)
description: mix using mixbox pigment algorithm https://github.com/scrtwpns/pigment-mixing and converted to GLSL by Patricio Gonzalez Vivo
use: <vec3\vec4> mixBox(<vec3|vec4> rgbA, <vec3|vec4> rgbB, float pct)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - MIXBOX_LUT: name of the texture uniform which you can find here https://github.com/scrtwpns/pigment-mixing
    - MIXBOX_LUT_FLIP_Y: when defined it expects a vertically flipled texture  
    - MIXBOX_LUT_SAMPLER_FNC: sampler function. Default: texture2D(MIXBOX_LUT, POS_UV).rgb
    - MIXBOX_LUT_CELL_SIZE: Default 256
license: |
    Copyright (c) 2022, Secret Weapons. All rights reserved.
    This code is for non-commercial use only. It is provided for research and evaluation purposes.
    If you wish to obtain commercial license, please contact: mixbox@scrtwpns.com
*/

#ifndef SAMPLER_FNC
#define SAMPLER_FNC(TEX, UV) texture2D(TEX, UV)
#endif

#ifndef MIXBOX_LUT 
#define MIXBOX_LUT u_tex0
#endif

#ifndef MIXBOX_LUT_SAMPLER_FNC
#define MIXBOX_LUT_SAMPLER_FNC(POS_UV) SAMPLER_FNC(MIXBOX_LUT, POS_UV).rgb
#endif

#ifndef MIXBOX_LUT_CELL_SIZE
#define MIXBOX_LUT_CELL_SIZE 256.0
#endif

#ifndef FNC_MIXBOX
#define FNC_MIXBOX

#define MIXBOX_NUMLATENTS 7

struct mixBox_latent {
    float data[MIXBOX_NUMLATENTS];
};

vec3 mixBox_mix(vec4 c) {
    vec3 coefs[20];
    coefs[ 0] = vec3( 0.07717053, 0.02826978, 0.24832992);
    coefs[ 1] = vec3( 0.95912302, 0.80256528, 0.03561839);
    coefs[ 2] = vec3( 0.74683774, 0.04868586, 0.00000000);
    coefs[ 3] = vec3( 0.99518138, 0.99978149, 0.99704802);
    coefs[ 4] = vec3( 0.04819146, 0.83363781, 0.32515377);
    coefs[ 5] = vec3(-0.6814695, 1.46107803, 1.06980936);
    coefs[ 6] = vec3( 0.27058419,-0.1532487, 1.98735057);
    coefs[ 7] = vec3(0.80478189, 0.6709371, 0.184245);
    coefs[ 8] = vec3(-0.35031003, 1.37855826, 3.68865);
    coefs[ 9] = vec3( 1.05128046, 1.97815239, 2.82989073);
    coefs[10] = vec3( 3.21607125, 0.81270228, 1.03384539);
    coefs[11] = vec3( 2.78893374, 0.41565549,-0.04487295);
    coefs[12] = vec3( 3.02162577, 2.55374103, 0.32766114);
    coefs[13] = vec3( 2.95124691, 2.81201112, 1.17578442);
    coefs[14] = vec3( 2.82677043, 0.79933038, 1.81715262);
    coefs[15] = vec3( 2.99691099, 1.22593053, 1.80653661);
    coefs[16] = vec3( 1.87394106, 2.05027182,-0.29835996);
    coefs[17] = vec3( 2.56609566, 7.03428198, 0.62575374);
    coefs[18] = vec3( 4.08329484,-1.40408358, 2.14995522);
    coefs[19] = vec3( 6.00078678, 2.55552042, 1.90739502);

    vec4 cc = c*c; 
    vec2 c0 = c.x*c.yz;

    float weights[20];
    weights[ 0] = c[0]*cc[0];
    weights[ 1] = c[1]*cc[1];
    weights[ 2] = c[2]*cc[2];
    weights[ 3] = c[3]*cc[3];
    weights[ 4] = cc[0]*c[1];
    weights[ 5] = c0[0]*c[1];
    weights[ 6] = cc[0]*c[2];
    weights[ 7] = c0[1]*c[2];
    weights[ 8] = cc[0]*c[3];
    weights[ 9] = c[0]*cc[3];
    weights[10] = cc[1]*c[2];
    weights[11] = c[1]*cc[2];
    weights[12] = cc[1]*c[3];
    weights[13] = c[1]*cc[3];
    weights[14] = cc[2]*c[3];
    weights[15] = c[2]*cc[3];
    weights[16] = c0[0]*c[2];
    weights[17] = c0[0]*c[3];
    weights[18] = c0[1]*c[3];
    weights[19] = c[1]*c[2]*c[3];
  
    vec3 rgb = vec3(0.0);
    for(int j = 0; j < 20; j++)
        rgb += weights[j] * coefs[j];

    return rgb;
}

mixBox_latent mixBox_srgb2latent(vec3 rgb) {
    vec3 xyz = saturate(rgb) * (MIXBOX_LUT_CELL_SIZE);
    vec3 xyz_i = floor(xyz);
    vec3 t = xyz - xyz_i;

    float weights[8];
    weights[0] = (1.0-t.x)*(1.0-t.y)*(1.0-t.z);
    weights[1] = (    t.x)*(1.0-t.y)*(1.0-t.z);
    weights[2] = (1.0-t.x)*(    t.y)*(1.0-t.z);
    weights[3] = (    t.x)*(    t.y)*(1.0-t.z);
    weights[4] = (1.0-t.x)*(1.0-t.y)*(    t.z);
    weights[5] = (    t.x)*(1.0-t.y)*(    t.z);
    weights[6] = (1.0-t.x)*(    t.y)*(    t.z);
    weights[7] = (    t.x)*(    t.y)*(    t.z);

    vec4 c = vec4(0.0);
    vec2 lutResFactor = 1./vec2(MIXBOX_LUT_CELL_SIZE * 16.0);
    float cellsFactor = 1./16.0; 
    for (int j = 0; j<8; j++) {
        vec2 uv = vec2(0.0);
        uv.x = mod(xyz_i.b, 16.0) * MIXBOX_LUT_CELL_SIZE + xyz_i.r;
        uv.y = (xyz_i.b * cellsFactor) * MIXBOX_LUT_CELL_SIZE + xyz_i.g;
        uv *= lutResFactor;
        #ifndef MIXBOX_LUT_FLIP_Y
        uv.y = 1.0 - uv.y;
        #endif
        c.rgb += weights[j] * MIXBOX_LUT_SAMPLER_FNC(uv);
    }
    c.a = 1.0 - (c.r+c.g+c.b);

    vec3 mixrgb = mixBox_mix(c);
    mixBox_latent out_latent;
    out_latent.data[0] = c[0];
    out_latent.data[1] = c[1];
    out_latent.data[2] = c[2];
    out_latent.data[3] = c[3];
    out_latent.data[4] = rgb.r - mixrgb.r;
    out_latent.data[5] = rgb.g - mixrgb.g;
    out_latent.data[6] = rgb.b - mixrgb.b;
    return out_latent;
}

vec3 mixBox_mix(mixBox_latent latent) {
    return mixBox_mix( vec4(latent.data[0], latent.data[1], latent.data[2], latent.data[3]) );
}

mixBox_latent mixBox_mix(mixBox_latent A, mixBox_latent B, float t) {
    mixBox_latent C;
    for (int l = 0; l < MIXBOX_NUMLATENTS; ++l)
        C.data[l] = mix(A.data[l], B.data[l], t);
    return C;
}

vec3 mixBox_latent2srgb(mixBox_latent latent) {
    return saturate( mixBox_mix( latent ) + vec3(latent.data[4], latent.data[5], latent.data[6]) );
}

vec3 mixBox(vec3 rgbA, vec3 rgbB, float t) {
    mixBox_latent latA = mixBox_srgb2latent(rgbA);
    mixBox_latent latB = mixBox_srgb2latent(rgbB);
    mixBox_latent latC = mixBox_mix(latA, latB, t);
    return mixBox_latent2srgb(latC);
}

vec4 mixBox(vec4 colA, vec4 colB, float t) {
    return vec4( mixBox(colA.rgb, colB.rgb, t), mix(colA.a, colB.a, t) );
}
#endif
