#include "../../math/mmax.hlsl"
#include "../../sampler.hlsl"

/*
contributors: bacondither
description: Adaptive sharpening. For strength values between 0.3 <-> 2.0 are a reasonable range 
use: sharpen(<SAMPLER_TYPE> texture, <float2> st, <float2> renderSize [, float streanght])
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - SHARPENADAPTIVE_TYPE: defaults to float3
    - SHARPENDADAPTIVE_SAMPLER_FNC(TEX, UV): defaults to texture2D(TEX, UV).rgb
    - SHARPENADAPTIVE_ANIME: only darken edges. Defaults to false
*/

#ifndef SHARPENADAPTIVE_TYPE
#ifdef SHARPEN_TYPE
#define SHARPENADAPTIVE_TYPE SHARPEN_TYPE
#else
#define SHARPENADAPTIVE_TYPE float4
#endif
#endif

#ifndef SHARPENDADAPTIVE_SAMPLER_FNC
#ifdef SHARPEN_SAMPLER_FNC
#define SHARPENDADAPTIVE_SAMPLER_FNC(TEX, UV) SHARPEN_SAMPLER_FNC(TEX, UV)
#else
#define SHARPENDADAPTIVE_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif
#endif

#ifndef SHARPENADAPTIVE_ANIME
#define SHARPENADAPTIVE_ANIME      false    // Only darken edges
#endif

#ifndef FNC_SHARPENAPTIVE
#define FNC_SHARPENAPTIVE

// Soft limit, modified tanh approx
#define SHARPENADAPTIVE_SOFT_LIM(v,s)  ( saturate(abs(v/s)*(27.0 + pow(v/s, 2.0))/(27.0 + 9.0*pow(v/s, 2.0)))*s )

// Weighted power mean
#define SHARPENADAPTIVE_WPMEAN(a,b,w)  ( pow(w*pow(abs(a), 0.5) + abs(1.0-w)*pow(abs(b), 0.5), 2.0) )

// Get destination pixel values
#define SHARPENADAPTIVE_DXDY(val)   ( length(fwidth(val)) ) // edgemul = 2.2

#ifndef SHARPENADAPTIVE_CTRL
// #define SHARPENADAPTIVE_CTRL(RGB)   ( dot(RGB*RGB, float3(0.212655, 0.715158, 0.072187)) )
#define SHARPENADAPTIVE_CTRL(RGB) sharpendAdaptiveControl(RGB)
#endif

float sharpendAdaptiveControl(in float3 rgb) { return dot(rgb*rgb, float3(0.212655, 0.715158, 0.072187)); }
float sharpendAdaptiveControl(in float4 rgba) { return dot(rgba*rgba, float4(0.212655, 0.715158, 0.072187, 0.0)); }

#define SHARPENADAPTIVE_DIFF(pix)   ( abs(blur-c[pix]) )

SHARPENADAPTIVE_TYPE sharpenAdaptive(SAMPLER_TYPE tex, float2 st, float2 pixel, float strength) {

    //-------------------------------------------------------------------------------------------------
// Defined values under this row are "optimal" DO NOT CHANGE IF YOU DO NOT KNOW WHAT YOU ARE DOING!
    const float curveslope  = 0.5  ;                // Sharpening curve slope, high edge values

    const float L_overshoot = 0.003;                // Max light overshoot before compression [>0.001]
    const float L_compr_low = 0.167;                // Light compression, default (0.167=~6x)

    const float D_overshoot = 0.009;                // Max dark overshoot before compression [>0.001]
    const float D_compr_low = 0.250;                // Dark compression, default (0.250=4x)

    const float scale_lim   = 0.1  ;                // Abs max change before compression [>0.01]
    const float scale_cs    = 0.056;                // Compression slope above scale_lim


    // [                c22               ]
    // [           c24, c9,  c23          ]
    // [      c21, c1,  c2,  c3, c18      ]
    // [ c19, c10, c4,  c0,  c5, c11, c16 ]
    // [      c20, c6,  c7,  c8, c17      ]
    // [           c15, c12, c14          ]
    // [                c13               ]
    SHARPENADAPTIVE_TYPE c[25];
    c[0] = SHARPENDADAPTIVE_SAMPLER_FNC(tex, st + float2(0.0, 0.0) * pixel);
    c[1] = SHARPENDADAPTIVE_SAMPLER_FNC(tex, st + float2(-1., -1.) * pixel);
    c[2] = SHARPENDADAPTIVE_SAMPLER_FNC(tex, st + float2(0.0, -1.) * pixel);
    c[3] = SHARPENDADAPTIVE_SAMPLER_FNC(tex, st + float2(1.0, -1.) * pixel);
    c[4] = SHARPENDADAPTIVE_SAMPLER_FNC(tex, st + float2(-1., 1.0) * pixel);
    c[5] = SHARPENDADAPTIVE_SAMPLER_FNC(tex, st + float2(1.0, 0.0) * pixel);
    c[6] = SHARPENDADAPTIVE_SAMPLER_FNC(tex, st + float2(-1., 1.0) * pixel);
    c[7] = SHARPENDADAPTIVE_SAMPLER_FNC(tex, st + float2(0.0, 1.0) * pixel);
    c[8] = SHARPENDADAPTIVE_SAMPLER_FNC(tex, st + float2(1.0, 1.0) * pixel);
    c[9] = SHARPENDADAPTIVE_SAMPLER_FNC(tex, st + float2(0.0, -2.) * pixel);
    c[10] = SHARPENDADAPTIVE_SAMPLER_FNC(tex, st + float2(-2., 0.0) * pixel);
    c[11] = SHARPENDADAPTIVE_SAMPLER_FNC(tex, st + float2( 2., 0.0) * pixel);
    c[12] = SHARPENDADAPTIVE_SAMPLER_FNC(tex, st + float2( 0., 2.0) * pixel);
    c[13] = SHARPENDADAPTIVE_SAMPLER_FNC(tex, st + float2( 0., 3.0) * pixel);
    c[14] = SHARPENDADAPTIVE_SAMPLER_FNC(tex, st + float2( 1., 2.0) * pixel);
    c[15] = SHARPENDADAPTIVE_SAMPLER_FNC(tex, st + float2(-1., 2.0) * pixel);
    c[16] = SHARPENDADAPTIVE_SAMPLER_FNC(tex, st + float2( 3., 0.0) * pixel);
    c[17] = SHARPENDADAPTIVE_SAMPLER_FNC(tex, st + float2( 2., 1.0) * pixel);
    c[18] = SHARPENDADAPTIVE_SAMPLER_FNC(tex, st + float2( 2.,-1.0) * pixel);
    c[19] = SHARPENDADAPTIVE_SAMPLER_FNC(tex, st + float2(-3., 0.0) * pixel);
    c[20] = SHARPENDADAPTIVE_SAMPLER_FNC(tex, st + float2(-2., 1.0) * pixel);
    c[21] = SHARPENDADAPTIVE_SAMPLER_FNC(tex, st + float2(-2.,-1.0) * pixel);
    c[22] = SHARPENDADAPTIVE_SAMPLER_FNC(tex, st + float2( 0.,-3.0) * pixel);
    c[23] = SHARPENDADAPTIVE_SAMPLER_FNC(tex, st + float2( 1.,-2.0) * pixel);
    c[24] = SHARPENDADAPTIVE_SAMPLER_FNC(tex, st + float2(-1.,-2.0) * pixel);

    float e[13];
    e[0] = SHARPENADAPTIVE_DXDY(c[0]);
    e[1] = SHARPENADAPTIVE_DXDY(c[1]);
    e[2] = SHARPENADAPTIVE_DXDY(c[2]);
    e[3] = SHARPENADAPTIVE_DXDY(c[3]);
    e[4] = SHARPENADAPTIVE_DXDY(c[4]);

    e[5] = SHARPENADAPTIVE_DXDY(c[5]);
    e[6] = SHARPENADAPTIVE_DXDY(c[6]);
    e[7] = SHARPENADAPTIVE_DXDY(c[7]);
    e[8] = SHARPENADAPTIVE_DXDY(c[8]);
    e[9] = SHARPENADAPTIVE_DXDY(c[9]);

    e[10] = SHARPENADAPTIVE_DXDY(c[10]);
    e[11] = SHARPENADAPTIVE_DXDY(c[11]);
    e[12] = SHARPENADAPTIVE_DXDY(c[12]);

    // Blur, gauss 3x3
    SHARPENADAPTIVE_TYPE  blur   = (2.0 * (c[2]+c[4]+c[5]+c[7]) + (c[1]+c[3]+c[6]+c[8]) + 4.0 * c[0]) / 16.0;

    // Contrast compression, center = 0.5, scaled to 1/3
    float fu = -7.4/3.0;
    float c_comp = saturate(0.266666681 + 0.9*exp2(dot(blur, float4(fu, fu, fu, fu))));

    // Edge detection
    // Relative matrix weights
    // [          1          ]
    // [      4,  5,  4      ]
    // [  1,  5,  6,  5,  1  ]
    // [      4,  5,  4      ]
    // [          1          ]
    float edge = length( 1.38*SHARPENADAPTIVE_DIFF(0)
                       + 1.15*(SHARPENADAPTIVE_DIFF(2) + SHARPENADAPTIVE_DIFF(4) + SHARPENADAPTIVE_DIFF(5) + SHARPENADAPTIVE_DIFF(7))
                       + 0.92*(SHARPENADAPTIVE_DIFF(1) + SHARPENADAPTIVE_DIFF(3) + SHARPENADAPTIVE_DIFF(6) + SHARPENADAPTIVE_DIFF(8))
                       + 0.23*(SHARPENADAPTIVE_DIFF(9) + SHARPENADAPTIVE_DIFF(10) + SHARPENADAPTIVE_DIFF(11) + SHARPENADAPTIVE_DIFF(12)) ) * c_comp;

    float2 cs = float2(L_compr_low,  D_compr_low);

    // RGB to luma
    float luma[25];
    luma[0] = SHARPENADAPTIVE_CTRL(c[0]);
    luma[1] = SHARPENADAPTIVE_CTRL(c[1]);
    luma[2] = SHARPENADAPTIVE_CTRL(c[2]);
    luma[3] = SHARPENADAPTIVE_CTRL(c[3]);
    luma[4] = SHARPENADAPTIVE_CTRL(c[4]);
    luma[5] = SHARPENADAPTIVE_CTRL(c[5]);
    luma[6] = SHARPENADAPTIVE_CTRL(c[6]);
    luma[7] = SHARPENADAPTIVE_CTRL(c[7]);
    luma[8] = SHARPENADAPTIVE_CTRL(c[8]);
    luma[9] = SHARPENADAPTIVE_CTRL(c[9]);
    luma[10] = SHARPENADAPTIVE_CTRL(c[10]);
    luma[11] = SHARPENADAPTIVE_CTRL(c[11]);
    luma[12] = SHARPENADAPTIVE_CTRL(c[12]);
    luma[13] = SHARPENADAPTIVE_CTRL(c[13]);
    luma[14] = SHARPENADAPTIVE_CTRL(c[14]);
    luma[15] = SHARPENADAPTIVE_CTRL(c[15]);
    luma[16] = SHARPENADAPTIVE_CTRL(c[16]);
    luma[17] = SHARPENADAPTIVE_CTRL(c[17]);
    luma[18] = SHARPENADAPTIVE_CTRL(c[18]);
    luma[19] = SHARPENADAPTIVE_CTRL(c[19]);
    luma[20] = SHARPENADAPTIVE_CTRL(c[20]);
    luma[21] = SHARPENADAPTIVE_CTRL(c[21]);
    luma[22] = SHARPENADAPTIVE_CTRL(c[22]);
    luma[23] = SHARPENADAPTIVE_CTRL(c[23]);
    luma[24] = SHARPENADAPTIVE_CTRL(c[24]);


    float c0_Y = sqrt(luma[0]);

    // Precalculated default squared kernel weights
    const float3 w1 = float3(0.5,           1.0, 1.41421356237); // 0.25, 1.0, 2.0
    const float3 w2 = float3(0.86602540378, 1.0, 0.54772255751); // 0.75, 1.0, 0.3

    // Transition to a concave kernel if the center edge val is above thr
    float3 dW = pow(lerp( w1, w2, saturate(2.4*edge - 0.82)), float3(2.0, 2.0, 2.0));

    // Use lower weights for pixels in a more active area relative to center pixel area
    // This results in narrower and less visible overshoots around sharp edges
    float modif_e0 = 3.0 * e[0] + 0.0090909;

    float weights[12];
    weights[0] = min(modif_e0/e[1],  dW.y);
    weights[1] = dW.x;
    weights[2] = min(modif_e0/e[3],  dW.y);
    weights[3] = dW.x;
    weights[4] = dW.x;
    weights[5] = min(modif_e0/e[6],  dW.y);
    weights[6] = dW.x;
    weights[7] = min(modif_e0/e[8],  dW.y);
    weights[8] = min(modif_e0/e[9],  dW.z);
    weights[9] = min(modif_e0/e[10], dW.z);
    weights[10] = min(modif_e0/e[11], dW.z);
    weights[11] = min(modif_e0/e[12], dW.z);

    weights[0] = (max(max((weights[8]  + weights[9])/4.0,  weights[0]), 0.25) + weights[0])/2.0;
    weights[2] = (max(max((weights[8]  + weights[10])/4.0, weights[2]), 0.25) + weights[2])/2.0;
    weights[5] = (max(max((weights[9]  + weights[11])/4.0, weights[5]), 0.25) + weights[5])/2.0;
    weights[7] = (max(max((weights[10] + weights[11])/4.0, weights[7]), 0.25) + weights[7])/2.0;

    // Calculate the negative part of the laplace kernel and the low threshold weight
    float lowthrsum   = 0.0;
    float weightsum   = 0.0;
    float neg_laplace = 0.0;

    for (int pix = 0; pix < 12; ++pix) {
        float lowthr = clamp((29.04*e[pix + 1] - 0.221), 0.01, 1.0);

        neg_laplace += luma[pix+1] * weights[pix] * lowthr;
        weightsum   += weights[pix] * lowthr;
        lowthrsum   += lowthr / 12.0;
    }

    neg_laplace = rsqrt(weightsum / neg_laplace);

    // Compute sharpening magnitude function
    float sharpen_val = strength/(strength*curveslope*pow(edge, 3.5) + 0.625);

    // Calculate sharpening diff and scale
    float sharpdiff = (c0_Y - neg_laplace)*(lowthrsum*sharpen_val + 0.01);

    // Calculate local near min & max, partial sort
    float temp = 0.0;

    for (int i1 = 0; i1 < 24; i1 += 2) {
        temp = luma[i1];
        luma[i1]   = min(luma[i1], luma[i1+1]);
        luma[i1+1] = max(temp, luma[i1+1]);
    }

    for (int i2 = 24; i2 > 0; i2 -= 2) {
        temp = luma[0];
        luma[0]    = min(luma[0], luma[i2]);
        luma[i2]   = max(temp, luma[i2]);

        temp = luma[24];
        luma[24] = max(luma[24], luma[i2-1]);
        luma[i2-1] = min(temp, luma[i2-1]);
    }

    for (int i1 = 1; i1 < 24-1; i1 += 2) {
        temp = luma[i1];
        luma[i1]   = min(luma[i1], luma[i1+1]);
        luma[i1+1] = max(temp, luma[i1+1]);
    }

    for (int i2 = 24-1; i2 > 1; i2 -= 2) {
        temp = luma[1];
        luma[1]    = min(luma[1], luma[i2]);
        luma[i2]   = max(temp, luma[i2]);

        temp = luma[24-1];
        luma[24-1] = max(luma[24-1], luma[i2-1]);
        luma[i2-1] = min(temp, luma[i2-1]);
    }

    float nmax = (max(sqrt(luma[23]), c0_Y)*2.0 + sqrt(luma[24]))/3.0;
    float nmin = (min(sqrt(luma[1]),  c0_Y)*2.0 + sqrt(luma[0]))/3.0;

    float min_dist  = min(abs(nmax - c0_Y), abs(c0_Y - nmin));
    float pos_scale = min_dist + L_overshoot;
    float neg_scale = min_dist + D_overshoot;

    pos_scale = min(pos_scale, scale_lim*(1.0 - scale_cs) + pos_scale*scale_cs);
    neg_scale = min(neg_scale, scale_lim*(1.0 - scale_cs) + neg_scale*scale_cs);

    // Soft limited anti-ringing with tanh, SHARPENADAPTIVE_WPMEAN to control compression slope
    sharpdiff = (SHARPENADAPTIVE_ANIME ? 0. :
                SHARPENADAPTIVE_WPMEAN(max(sharpdiff, 0.0), SHARPENADAPTIVE_SOFT_LIM( max(sharpdiff, 0.0), pos_scale ), cs.x ))
              - SHARPENADAPTIVE_WPMEAN(min(sharpdiff, 0.0), SHARPENADAPTIVE_SOFT_LIM( min(sharpdiff, 0.0), neg_scale ), cs.y );

    float sharpdiff_lim = saturate(c0_Y + sharpdiff) - c0_Y;
    float satmul = (c0_Y + max(sharpdiff_lim*0.9, sharpdiff_lim)*1.03 + 0.03)/(c0_Y + 0.03);
    return c0_Y + (sharpdiff_lim*3.0 + sharpdiff)/4.0 + (c[0] - c0_Y)*satmul;
}

SHARPENADAPTIVE_TYPE sharpenAdaptive(SAMPLER_TYPE tex, float2 st, float2 pixel) {
    return sharpenAdaptive(tex, st, pixel, 1.0);
}

#endif