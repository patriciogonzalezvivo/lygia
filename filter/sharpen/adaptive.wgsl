
/*
contributors: bacondither
description: Adaptive sharpening. For strenght values between 0.3 <-> 2.0 are a reasonable range
use: sharpen(<SAMPLER_TYPE> texture, <vec2> st, <vec2> renderSize [, float streanght])
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - SHARPENADAPTIVE_TYPE: defaults to vec3
    - SHARPENDADAPTIVE_SAMPLER_FNC(TEX, UV): defaults to texture2D(TEX, UV).rgb
    - SHARPENADAPTIVE_ANIME: only darken edges. Defaults to false
examples:
    - /shaders/filter_sharpen2D.frag
*/

fn sharpenContrastAdaptive(myTexture
                           : texture_2d<f32>, mySampler
                           : sampler, st
                           : vec2f, pixel
                           : vec2f, strength
                           : f32) -> vec3f {
    let peak = -1.0 / mix(8.0, 5.0, saturate(strength));

    // fetch a 3x3 neighborhood around the pixel 'e',
    //  a b c
    //  d(e)f
    //  g h i
    let a = textureSampleBaseClampToEdge(myTexture, mySampler, st + vec2(-1., -1.) * pixel).rgb;
    let b = textureSampleBaseClampToEdge(myTexture, mySampler, st + vec2(0., -1.) * pixel).rgb;
    let c = textureSampleBaseClampToEdge(myTexture, mySampler, st + vec2(1., -1.) * pixel).rgb;
    let d = textureSampleBaseClampToEdge(myTexture, mySampler, st + vec2(-1., 0.) * pixel).rgb;
    let e = textureSampleBaseClampToEdge(myTexture, mySampler, st + vec2(0., 0.) * pixel).rgb;
    let f = textureSampleBaseClampToEdge(myTexture, mySampler, st + vec2(1., 0.) * pixel).rgb;
    let g = textureSampleBaseClampToEdge(myTexture, mySampler, st + vec2(-1., 1.) * pixel).rgb;
    let h = textureSampleBaseClampToEdge(myTexture, mySampler, st + vec2(0., 1.) * pixel).rgb;
    let i = textureSampleBaseClampToEdge(myTexture, mySampler, st + vec2(1., 1.) * pixel).rgb;

    // Soft min and max.
    //  a b c             b
    //  d e f * 0.5  +  d e f * 0.5
    //  g h i             h
    // These are 2.0x bigger (factored out the extra multiply).
    var mnRGB = min(min(min(d, e), min(f, b)), h);
    let mnRGB2 = min(mnRGB, min(min(a, c), min(g, i)));
    mnRGB += mnRGB2;

    var mxRGB = max(max(max(d, e), max(f, b)), h);
    let mxRGB2 = max(mxRGB, max(max(a, c), max(g, i)));
    mxRGB += mxRGB2;

    // Smooth minimum distance to signal limit divided by smooth max.
    let ampRGB = saturate(min(mnRGB, 2.0 - mxRGB) / mxRGB);

    // Shaping amount of sharpening.
    let wRGB = sqrt(ampRGB) * peak;

    // Filter shape.
    //  0 w 0
    //  w 1 w
    //  0 w 0
    let weightRGB = 1.0 + 4.0 * wRGB;
    let window = (b + d) + (f + h);

    return saturate((window * wRGB + e) / weightRGB);
}

const SHARPENADAPTIVE_ANIME = false;  // Only darken edges

// Soft limit, modified tanh approx
fn SHARPENADAPTIVE_SOFT_LIM(v : f32, s : f32) -> f32 {
    return (saturate(abs(v / s) * (27.0 + pow(v / s, 2.0)) / (27.0 + 9.0 * pow(v / s, 2.0))) * s);
}

// Weighted power mean
fn SHARPENADAPTIVE_WPMEAN(a : f32, b : f32, w : f32) -> f32 {
    return (pow(w * pow(abs(a), 0.5) + abs(1.0 - w) * pow(abs(b), 0.5), 2.0));
}

// Get destination pixel values
fn SHARPENADAPTIVE_DXDY(val : vec4f) -> f32 {
    return (length(fwidth(val)));  // edgemul = 2.2
}

// #define SHARPENADAPTIVE_CTRL(RGB)   ( dot(RGB*RGB, vec3(0.212655, 0.715158, 0.072187)) )
fn SHARPENADAPTIVE_CTRL(RGB : vec4f) -> f32 { return sharpendAdaptiveControl4(RGB); }

fn sharpendAdaptiveControl3(rgb : vec3f) -> f32 { return dot(rgb * rgb, vec3(0.212655, 0.715158, 0.072187)); }
fn sharpendAdaptiveControl4(rgba : vec4f) -> f32 { return dot(rgba * rgba, vec4(0.212655, 0.715158, 0.072187, 0.0)); }

//-------------------------------------------------------------------------------------------------
// Defined values under this row are "optimal" DO NOT CHANGE IF YOU DO NOT KNOW WHAT YOU ARE DOING!
const curveslope = 0.5;  // Sharpening curve slope, high edge values

const L_overshoot = 0.003;  // Max light overshoot before compression [>0.001]
const L_compr_low = 0.167;  // Light compression, default (0.167=~6x)

const D_overshoot = 0.009;  // Max dark overshoot before compression [>0.001]
const D_compr_low = 0.250;  // Dark compression, default (0.250=4x)

const scale_lim = 0.1;   // Abs max change before compression [>0.01]
const scale_cs = 0.056;  // Compression slope above scale_lim

// Precalculated default squared kernel weights
const w1 = vec3(0.5, 1.0, 1.41421356237);            // 0.25, 1.0, 2.0
const w2 = vec3(0.86602540378, 1.0, 0.54772255751);  // 0.75, 1.0, 0.3

fn sharpenAdaptive(myTexture
                   : texture_2d<f32>, mySampler
                   : sampler, st
                   : vec2f, pixel
                   : vec2f, strength
                   : f32) -> vec4f {
    // [                c22               ]
    // [           c24, c9,  c23          ]
    // [      c21, c1,  c2,  c3, c18      ]
    // [ c19, c10, c4,  c0,  c5, c11, c16 ]
    // [      c20, c6,  c7,  c8, c17      ]
    // [           c15, c12, c14          ]
    // [                c13               ]
    let c = array<vec4f, 25>(textureSampleBaseClampToEdge(myTexture, mySampler, st + vec2(0.0, 0.0) * pixel),
                             textureSampleBaseClampToEdge(myTexture, mySampler, st + vec2(-1., -1.) * pixel),
                             textureSampleBaseClampToEdge(myTexture, mySampler, st + vec2(0.0, -1.) * pixel),
                             textureSampleBaseClampToEdge(myTexture, mySampler, st + vec2(1.0, -1.) * pixel),
                             textureSampleBaseClampToEdge(myTexture, mySampler, st + vec2(-1., 1.0) * pixel),
                             textureSampleBaseClampToEdge(myTexture, mySampler, st + vec2(1.0, 0.0) * pixel),
                             textureSampleBaseClampToEdge(myTexture, mySampler, st + vec2(-1., 1.0) * pixel),
                             textureSampleBaseClampToEdge(myTexture, mySampler, st + vec2(0.0, 1.0) * pixel),
                             textureSampleBaseClampToEdge(myTexture, mySampler, st + vec2(1.0, 1.0) * pixel),
                             textureSampleBaseClampToEdge(myTexture, mySampler, st + vec2(0.0, -2.) * pixel),
                             textureSampleBaseClampToEdge(myTexture, mySampler, st + vec2(-2., 0.0) * pixel),
                             textureSampleBaseClampToEdge(myTexture, mySampler, st + vec2(2., 0.0) * pixel),
                             textureSampleBaseClampToEdge(myTexture, mySampler, st + vec2(0., 2.0) * pixel),
                             textureSampleBaseClampToEdge(myTexture, mySampler, st + vec2(0., 3.0) * pixel),
                             textureSampleBaseClampToEdge(myTexture, mySampler, st + vec2(1., 2.0) * pixel),
                             textureSampleBaseClampToEdge(myTexture, mySampler, st + vec2(-1., 2.0) * pixel),
                             textureSampleBaseClampToEdge(myTexture, mySampler, st + vec2(3., 0.0) * pixel),
                             textureSampleBaseClampToEdge(myTexture, mySampler, st + vec2(2., 1.0) * pixel),
                             textureSampleBaseClampToEdge(myTexture, mySampler, st + vec2(2., -1.0) * pixel),
                             textureSampleBaseClampToEdge(myTexture, mySampler, st + vec2(-3., 0.0) * pixel),
                             textureSampleBaseClampToEdge(myTexture, mySampler, st + vec2(-2., 1.0) * pixel),
                             textureSampleBaseClampToEdge(myTexture, mySampler, st + vec2(-2., -1.0) * pixel),
                             textureSampleBaseClampToEdge(myTexture, mySampler, st + vec2(0., -3.0) * pixel),
                             textureSampleBaseClampToEdge(myTexture, mySampler, st + vec2(1., -2.0) * pixel),
                             textureSampleBaseClampToEdge(myTexture, mySampler, st + vec2(-1., -2.0) * pixel));

    let e = array<f32, 13>(SHARPENADAPTIVE_DXDY(c[0]), SHARPENADAPTIVE_DXDY(c[1]), SHARPENADAPTIVE_DXDY(c[2]),
                           SHARPENADAPTIVE_DXDY(c[3]), SHARPENADAPTIVE_DXDY(c[4]), SHARPENADAPTIVE_DXDY(c[5]),
                           SHARPENADAPTIVE_DXDY(c[6]), SHARPENADAPTIVE_DXDY(c[7]), SHARPENADAPTIVE_DXDY(c[8]),
                           SHARPENADAPTIVE_DXDY(c[9]), SHARPENADAPTIVE_DXDY(c[10]), SHARPENADAPTIVE_DXDY(c[11]),
                           SHARPENADAPTIVE_DXDY(c[12]));

    // Blur, gauss 3x3
    let blur = (2.0 * (c[2] + c[4] + c[5] + c[7]) + (c[1] + c[3] + c[6] + c[8]) + 4.0 * c[0]) / 16.0;

    // Contrast compression, center = 0.5, scaled to 1/3
    let c_comp = saturate(0.266666681 + 0.9 * exp2(dot(blur, vec4f(-7.4 / 3.0))));

    // Edge detection
    // Relative matrix weights
    // [          1          ]
    // [      4,  5,  4      ]
    // [  1,  5,  6,  5,  1  ]
    // [      4,  5,  4      ]
    // [          1          ]
    /*
     *fn SHARPENADAPTIVE_DIFF(pix : f32) -> f32 {
     *    return (abs(blur - c[pix]));
     *}
     */
    let edge = length(1.38 * (abs(blur - c[0])) +
                      1.15 * ((abs(blur - c[2])) + (abs(blur - c[4])) + (abs(blur - c[5])) + (abs(blur - c[7]))) +
                      0.92 * ((abs(blur - c[1])) + (abs(blur - c[3])) + (abs(blur - c[6])) + (abs(blur - c[8]))) +
                      0.23 * ((abs(blur - c[9])) + (abs(blur - c[10])) + (abs(blur - c[11])) + (abs(blur - c[12])))) *
               c_comp;

    let cs = vec2(L_compr_low, D_compr_low);

    // RGB to luma
    var luma = array<f32, 25>(SHARPENADAPTIVE_CTRL(c[0]), SHARPENADAPTIVE_CTRL(c[1]), SHARPENADAPTIVE_CTRL(c[2]),
                              SHARPENADAPTIVE_CTRL(c[3]), SHARPENADAPTIVE_CTRL(c[4]), SHARPENADAPTIVE_CTRL(c[5]),
                              SHARPENADAPTIVE_CTRL(c[6]), SHARPENADAPTIVE_CTRL(c[7]), SHARPENADAPTIVE_CTRL(c[8]),
                              SHARPENADAPTIVE_CTRL(c[9]), SHARPENADAPTIVE_CTRL(c[10]), SHARPENADAPTIVE_CTRL(c[11]),
                              SHARPENADAPTIVE_CTRL(c[12]), SHARPENADAPTIVE_CTRL(c[13]), SHARPENADAPTIVE_CTRL(c[14]),
                              SHARPENADAPTIVE_CTRL(c[15]), SHARPENADAPTIVE_CTRL(c[16]), SHARPENADAPTIVE_CTRL(c[17]),
                              SHARPENADAPTIVE_CTRL(c[18]), SHARPENADAPTIVE_CTRL(c[19]), SHARPENADAPTIVE_CTRL(c[20]),
                              SHARPENADAPTIVE_CTRL(c[21]), SHARPENADAPTIVE_CTRL(c[22]), SHARPENADAPTIVE_CTRL(c[23]),
                              SHARPENADAPTIVE_CTRL(c[24]));

    let c0_Y = sqrt(luma[0]);

    // Transition to a concave kernel if the center edge val is above thr
    let dW = pow(mix(w1, w2, saturate(2.4 * edge - 0.82)), vec3(2.0));

    // Use lower weights for pixels in a more active area relative to center pixel area
    // This results in narrower and less visible overshoots around sharp edges
    let modif_e0 = 3.0 * e[0] + 0.0090909;

    var weights =
        array<f32, 12>(min(modif_e0 / e[1], dW.y), dW.x, min(modif_e0 / e[3], dW.y), dW.x, dW.x,
                       min(modif_e0 / e[6], dW.y), dW.x, min(modif_e0 / e[8], dW.y), min(modif_e0 / e[9], dW.z),
                       min(modif_e0 / e[10], dW.z), min(modif_e0 / e[11], dW.z), min(modif_e0 / e[12], dW.z));

    weights[0] = (max(max((weights[8] + weights[9]) / 4.0, weights[0]), 0.25) + weights[0]) / 2.0;
    weights[2] = (max(max((weights[8] + weights[10]) / 4.0, weights[2]), 0.25) + weights[2]) / 2.0;
    weights[5] = (max(max((weights[9] + weights[11]) / 4.0, weights[5]), 0.25) + weights[5]) / 2.0;
    weights[7] = (max(max((weights[10] + weights[11]) / 4.0, weights[7]), 0.25) + weights[7]) / 2.0;

    // Calculate the negative part of the laplace kernel and the low threshold weight
    var lowthrsum = 0.0;
    var weightsum = 0.0;
    var neg_laplace = 0.0;

    for (var pix = 0; pix < 12; pix += 1) {
        let lowthr = clamp((29.04 * e[pix + 1] - 0.221), 0.01, 1.0);

        neg_laplace += luma[pix + 1] * weights[pix] * lowthr;
        weightsum += weights[pix] * lowthr;
        lowthrsum += lowthr / 12.0;
    }

    neg_laplace = inverseSqrt(weightsum / neg_laplace);

    // Compute sharpening magnitude function
    let sharpen_val = strength / (strength * curveslope * pow(edge, 3.5) + 0.625);

    // Calculate sharpening diff and scale
    var sharpdiff = (c0_Y - neg_laplace) * (lowthrsum * sharpen_val + 0.01);

    // Calculate local near min & max, partial sort
    var temp = 0.0;

    for (var i1 = 0; i1 < 24; i1 += 2) {
        temp = luma[i1];
        luma[i1] = min(luma[i1], luma[i1 + 1]);
        luma[i1 + 1] = max(temp, luma[i1 + 1]);
    }

    for (var i2 = 24; i2 > 0; i2 -= 2) {
        temp = luma[0];
        luma[0] = min(luma[0], luma[i2]);
        luma[i2] = max(temp, luma[i2]);

        temp = luma[24];
        luma[24] = max(luma[24], luma[i2 - 1]);
        luma[i2 - 1] = min(temp, luma[i2 - 1]);
    }

    for (var i1 = 1; i1 < 24 - 1; i1 += 2) {
        temp = luma[i1];
        luma[i1] = min(luma[i1], luma[i1 + 1]);
        luma[i1 + 1] = max(temp, luma[i1 + 1]);
    }

    for (var i2 = 24 - 1; i2 > 1; i2 -= 2) {
        temp = luma[1];
        luma[1] = min(luma[1], luma[i2]);
        luma[i2] = max(temp, luma[i2]);

        temp = luma[24 - 1];
        luma[24 - 1] = max(luma[24 - 1], luma[i2 - 1]);
        luma[i2 - 1] = min(temp, luma[i2 - 1]);
    }

    let nmax = (max(sqrt(luma[23]), c0_Y) * 2.0 + sqrt(luma[24])) / 3.0;
    let nmin = (min(sqrt(luma[1]), c0_Y) * 2.0 + sqrt(luma[0])) / 3.0;

    let min_dist = min(abs(nmax - c0_Y), abs(c0_Y - nmin));
    var pos_scale = min_dist + L_overshoot;
    var neg_scale = min_dist + D_overshoot;

    pos_scale = min(pos_scale, scale_lim * (1.0 - scale_cs) + pos_scale * scale_cs);
    neg_scale = min(neg_scale, scale_lim * (1.0 - scale_cs) + neg_scale * scale_cs);

    // Soft limited anti-ringing with tanh, SHARPENADAPTIVE_WPMEAN to control compression slope

    if (SHARPENADAPTIVE_ANIME) {
        sharpdiff = 0;
    } else {
        sharpdiff =
            SHARPENADAPTIVE_WPMEAN(max(sharpdiff, 0.0), SHARPENADAPTIVE_SOFT_LIM(max(sharpdiff, 0.0), pos_scale), cs.x);
    }

    sharpdiff -=
        SHARPENADAPTIVE_WPMEAN(min(sharpdiff, 0.0), SHARPENADAPTIVE_SOFT_LIM(min(sharpdiff, 0.0), neg_scale), cs.y);

    let sharpdiff_lim = saturate(c0_Y + sharpdiff) - c0_Y;
    let satmul = (c0_Y + max(sharpdiff_lim * 0.9, sharpdiff_lim) * 1.03 + 0.03) / (c0_Y + 0.03);
    return c0_Y + (sharpdiff_lim * 3.0 + sharpdiff) / 4.0 + (c[0] - c0_Y) * satmul;
}