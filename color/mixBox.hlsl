/*
original_author: Secret Weapons (@scrtwpns)
description: mix using mixbox pigment algorithm https://github.com/scrtwpns/pigment-mixing and converted to GLSL by Patricio Gonzalez Vivo
use: <float3\float4> mixBox(<float3|float4> rgbA, <float3|float4> rgbB, float pct)
options:
    - MIXBOX_LUT: name of the texture uniform which you can find here https://github.com/scrtwpns/pigment-mixing
    - MIXBOX_LUT_FLIP_Y: when defined it expects a vertically flipled texture  
    - MIXBOX_LUT_SAMPLER_FNC: sampler function. Default: texture2D(MIXBOX_LUT, POS_UV).rgb

license: |
    Copyright (c) 2022, Secret Weapons. All rights reserved.
    This code is for non-commercial use only. It is provided for research and evaluation purposes.
    If you wish to obtain commercial license, please contact: mixbox@scrtwpns.com
*/

#ifndef FNC_MIXBOX
#define FNC_MIXBOX

#ifndef MIXBOX_LUT 
#define MIXBOX_LUT u_tex0
#endif

#ifndef MIXBOX_LUT_SAMPLER_FNC
#define MIXBOX_LUT_SAMPLER_FNC(POS_UV) tex2D(MIXBOX_LUT, POS_UV).rgb
#endif

#define MIXBOX_NUMLATENTS 7

float3 mixBox_mix(float4 c) {
    float3 coefs[20];
    coefs[ 0] = float3(1.0* 0.07717053,1.0* 0.02826978,1.0* 0.24832992);
    coefs[ 1] = float3(1.0* 0.95912302,1.0* 0.80256528,1.0* 0.03561839);
    coefs[ 2] = float3(1.0* 0.74683774,1.0* 0.04868586,1.0* 0.00000000);
    coefs[ 3] = float3(1.0* 0.99518138,1.0* 0.99978149,1.0* 0.99704802);
    coefs[ 4] = float3(3.0* 0.01606382,3.0* 0.27787927,3.0* 0.10838459);
    coefs[ 5] = float3(3.0*-0.22715650,3.0* 0.48702601,3.0* 0.35660312);
    coefs[ 6] = float3(3.0* 0.09019473,3.0*-0.05108290,3.0* 0.66245019);
    coefs[ 7] = float3(3.0* 0.26826063,3.0* 0.22364570,3.0* 0.06141500);
    coefs[ 8] = float3(3.0*-0.11677001,3.0* 0.45951942,3.0* 1.22955000);
    coefs[ 9] = float3(3.0* 0.35042682,3.0* 0.65938413,3.0* 0.94329691);
    coefs[10] = float3(3.0* 1.07202375,3.0* 0.27090076,3.0* 0.34461513);
    coefs[11] = float3(3.0* 0.92964458,3.0* 0.13855183,3.0*-0.01495765);
    coefs[12] = float3(3.0* 1.00720859,3.0* 0.85124701,3.0* 0.10922038);
    coefs[13] = float3(3.0* 0.98374897,3.0* 0.93733704,3.0* 0.39192814);
    coefs[14] = float3(3.0* 0.94225681,3.0* 0.26644346,3.0* 0.60571754);
    coefs[15] = float3(3.0* 0.99897033,3.0* 0.40864351,3.0* 0.60217887);
    coefs[16] = float3(6.0* 0.31232351,6.0* 0.34171197,6.0*-0.04972666);
    coefs[17] = float3(6.0* 0.42768261,6.0* 1.17238033,6.0* 0.10429229);
    coefs[18] = float3(6.0* 0.68054914,6.0*-0.23401393,6.0* 0.35832587);
    coefs[19] = float3(6.0* 1.00013113,6.0* 0.42592007,6.0* 0.31789917);

    float3 rgb = float3(0.0, 0.0, 0.0);
    float c00 = c[0]*c[0];
    float c11 = c[1]*c[1];
    float c22 = c[2]*c[2];
    float c33 = c[3]*c[3];
    float c01 = c[0]*c[1];
    float c02 = c[0]*c[2];

    float weights[20];
    weights[ 0] = c[0]*c00;
    weights[ 1] = c[1]*c11;
    weights[ 2] = c[2]*c22;
    weights[ 3] = c[3]*c33;
    weights[ 4] = c00*c[1];
    weights[ 5] = c01*c[1];
    weights[ 6] = c00*c[2];
    weights[ 7] = c02*c[2];
    weights[ 8] = c00*c[3];
    weights[ 9] = c[0]*c33;
    weights[10] = c11*c[2];
    weights[11] = c[1]*c22;
    weights[12] = c11*c[3];
    weights[13] = c[1]*c33;
    weights[14] = c22*c[3];
    weights[15] = c[2]*c33;
    weights[16] = c01*c[2];
    weights[17] = c01*c[3];
    weights[18] = c02*c[3];
    weights[19] = c[1]*c[2]*c[3];
  
    for(int j=0;j<20;j++)
        for(int i=0;i<3;i++)
            rgb[i] += weights[j] * coefs[j][i];

    return rgb;
}

struct mixBox_latent {
    float data[MIXBOX_NUMLATENTS];
};

mixBox_latent mixBox_srgb_to_latent(float3 rgb) {
    float3 xyz = saturate(rgb) * 255.0;
    float3 xyz_i = floor(xyz);
    float3 t = xyz - xyz_i;

    float weights[8];
    weights[0] = (1.0-t.x)*(1.0-t.y)*(1.0-t.z);
    weights[1] = (    t.x)*(1.0-t.y)*(1.0-t.z);
    weights[2] = (1.0-t.x)*(    t.y)*(1.0-t.z);
    weights[3] = (    t.x)*(    t.y)*(1.0-t.z);
    weights[4] = (1.0-t.x)*(1.0-t.y)*(    t.z);
    weights[5] = (    t.x)*(1.0-t.y)*(    t.z);
    weights[6] = (1.0-t.x)*(    t.y)*(    t.z);
    weights[7] = (    t.x)*(    t.y)*(    t.z);

    float4 c = float4(0.0, 0.0, 0.0, 0.0);
    vec2 lutRes = 1./vec2(4096.0, 4096.0);
    for (int j = 0; j<8; j++) {
        vec2 uv = vec2(0.0);
        uv.x = mod(xyz_i.b, 16.0) * 256.0 + xyz_i.r;
        uv.y = (xyz_i.b / 16.0) * 256.0 + xyz_i.g;
        uv *= lutRes;
        #ifndef MIXBOX_LUT_FLIP_Y
        uv.y = 1.0 - uv.y;
        #endif
        c.rgb += weights[j] * MIXBOX_LUT_SAMPLER_FNC(uv);
    }

    c[3] = 1.0 - (c[0]+c[1]+c[2]);

    float3 mixrgb = mixBox_mix(c);
    mixBox_latent out_latent;
    out_latent.data[0] = c[0];
    out_latent.data[1] = c[1];
    out_latent.data[2] = c[2];
    out_latent.data[3] = c[3];
    out_latent.data[4] = (rgb.r - mixrgb[0]);
    out_latent.data[5] = (rgb.g - mixrgb[1]);
    out_latent.data[6] = (rgb.b - mixrgb[2]);
    return out_latent;
}

float3 mixBox_mix(mixBox_latent latent) {
    return mixBox_mix( float4(latent.data[0], latent.data[1], latent.data[2], latent.data[3]) );
}

float3 mixBox_latent_to_srgb(mixBox_latent latent) {
    float3 rgb = mixBox_mix( latent );
    return saturate(rgb + float3(latent.data[4], latent.data[5], latent.data[6]) );
}

float3 mixBox(float3 rgb1, float3 rgb2, float t) {
    mixBox_latent latent_A;
    mixBox_latent latent_B;

    latent_A = mixBox_srgb_to_latent(rgb1);
    latent_B = mixBox_srgb_to_latent(rgb2);
    
    mixBox_latent latent_C;

    for (int l = 0; l < MIXBOX_NUMLATENTS; ++l)
        latent_C.data[l] = (1.0-t) * latent_A.data[l] + t*latent_B.data[l];

    return ( mixBox_latent_to_srgb(latent_C) );
}

float4 mixBox(float4 colA, float4 colB, float t) {
    return float4( mixBox(colA.rgb, colB.rgb, t), lerp(colA.a, colB.a, t) );
}

#endif
