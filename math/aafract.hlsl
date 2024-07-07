/*
contributors: Fabrice Neyret
description: |
    Anti-aliasing fract function, including Shannon-Nyquiest filtering for high frequencies.
    Based on this example https://www.shadertoy.com/view/3tSGWy
use: <float> aafract(<float> x)
option:
    AA_EDGE: in the absence of derivatives you can specify the antialiasing factor
*/

#ifndef FNC_AAFRACT
#define FNC_AAFRACT

#ifndef NYQUIST_FNC
#define NYQUIST_BIAS -.0  // < 0: prefer a bit of aliasing to blur 
#define NYQUIST_SPREAD 1. // < 1: transition more brutal 
// w = pixel width = grad(continous signal) . c = possibly fracted signal.
#define NYQUIST_FNC(w,c) lerp(.5, c, clamp((.5-NYQUIST_BIAS-(w))/.25/NYQUIST_SPREAD,0.,1.) )
#endif

float aafract(float x) {
#if defined(AA_EDGE)
    float v = frac(x),
          w = AA_EDGE,      // pixel width. NB: x must not be discontinuous or factor discont out
          c = v < 1.-w ? v/(1.-w) : (1.-v)/w; // replace right step by down slope (-> chainsaw is continuous).
               // shortened slope : added downslope near v=1 
    return NYQUIST_FNC(w,c);
#else 
    float v = frac(x),
          w = length(float2(ddx(x), ddy(x))), // pixel width. NB: x must not be discontinuous or factor discont out
          c = v < 1. - w ? v / (1. - w) : (1. - v) / w; // replace right step by down slope (-> chainsaw is continuous).
               // shortened slope : added downslope near v=1 
    return NYQUIST_FNC(w, c);
#endif
}

float2 aafract(float2 v) {
    return float2(aafract(v.x), aafract(v.y));
}

#endif