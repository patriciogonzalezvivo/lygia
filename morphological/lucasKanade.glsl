#include "../sampler.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    Dense per-pixel optical flow between two samples of the same scalar field
    (e.g. two consecutive frames' luminance, or two depth buffers), solved via
    a windowed Lucas-Kanade least-squares fit over the spatio-temporal
    gradient structure tensor. Returns displacement in texel units (multiply
    by the UV step to get a UV offset). Assumes small motion between the two
    samples -- no coarse-to-fine pyramid here -- so it's blind to displacement
    much larger than LUCASKANADE_RADIUS texels/frame.
use: lucasKanade(<SAMPLER_TYPE> currTex, <SAMPLER_TYPE> prevTex, <vec2> st, <vec2> pixel)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - LUCASKANADE_SAMPLE_FNC(TEX, UV): which scalar field to track -- defaults
      to the texture's own red channel (a single-channel buffer, e.g. depth);
      pass a custom one (e.g. luma(SAMPLER_FNC(TEX, UV).rgb)) to track a color
      image's luminance instead
    - LUCASKANADE_RADIUS: window radius in texels aggregated per pixel
      (default 1, a 3x3 neighborhood) -- bigger is smoother/more robust to
      noise but blind to motion faster than ~this many texels/frame
    - LUCASKANADE_EPSILON: Tikhonov regularization epsilon added to the
      structure tensor's diagonal so flat/low-texture regions -- where the
      system is singular, the classic optical-flow aperture problem --
      settle to ~zero flow instead of dividing by a near-zero determinant
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef LUCASKANADE_SAMPLE_FNC
#define LUCASKANADE_SAMPLE_FNC(TEX, UV) SAMPLER_FNC(TEX, UV).r
#endif

#ifndef LUCASKANADE_RADIUS
#define LUCASKANADE_RADIUS 1
#endif

#ifndef LUCASKANADE_EPSILON
#define LUCASKANADE_EPSILON 0.001
#endif

#ifndef FNC_LUCASKANADE
#define FNC_LUCASKANADE
vec2 lucasKanade(SAMPLER_TYPE currTex, SAMPLER_TYPE prevTex, vec2 st, vec2 pixel) {
    float Sxx = 0.0, Syy = 0.0, Sxy = 0.0, Sxt = 0.0, Syt = 0.0;

    for (int j = -LUCASKANADE_RADIUS; j <= LUCASKANADE_RADIUS; j++)
    for (int i = -LUCASKANADE_RADIUS; i <= LUCASKANADE_RADIUS; i++) {
        vec2 uv = st + vec2(float(i), float(j)) * pixel;

        float cL = LUCASKANADE_SAMPLE_FNC(currTex, uv - vec2(pixel.x, 0.0));
        float cR = LUCASKANADE_SAMPLE_FNC(currTex, uv + vec2(pixel.x, 0.0));
        float cD = LUCASKANADE_SAMPLE_FNC(currTex, uv - vec2(0.0, pixel.y));
        float cU = LUCASKANADE_SAMPLE_FNC(currTex, uv + vec2(0.0, pixel.y));

        // Span of exactly one texel each side (no /pixel normalization) --
        // what makes the solved (u,v) come out already in texel units
        // rather than per-normalized-uv units.
        float Ix = (cR - cL) * 0.5;
        float Iy = (cU - cD) * 0.5;
        float It = LUCASKANADE_SAMPLE_FNC(currTex, uv) - LUCASKANADE_SAMPLE_FNC(prevTex, uv);

        Sxx += Ix * Ix;
        Syy += Iy * Iy;
        Sxy += Ix * Iy;
        Sxt += Ix * It;
        Syt += Iy * It;
    }

    Sxx += LUCASKANADE_EPSILON;
    Syy += LUCASKANADE_EPSILON;

    // Solve [[Sxx,Sxy],[Sxy,Syy]] * (u,v) = -(Sxt,Syt) via Cramer's rule.
    float det = Sxx * Syy - Sxy * Sxy;
    return vec2(Sxy * Syt - Syy * Sxt,
                Sxy * Sxt - Sxx * Syt) / det;
}
#endif
