/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Exponential easing. From https://github.com/stackgl/glsl-easings
use: exponential<In|Out|InOut>(<float> x)
*/

#ifndef FNC_EXPONENTIALIN
#define FNC_EXPONENTIALIN
float exponentialIn(in float t) {
    return t == 0.0 ? t : pow(2.0, 10.0 * (t - 1.0));
}
#endif

#ifndef FNC_EXPONENTIALOUT
#define FNC_EXPONENTIALOUT
float exponentialOut(in float t) {
    return t == 1.0 ? t : 1.0 - pow(2.0, -10.0 * t);
}
#endif

#ifndef FNC_EXPONENTIALINOUT
#define FNC_EXPONENTIALINOUT
float exponentialInOut(in float t) {
    return t == 0.0 || t == 1.0
        ? t
        : t < 0.5
            ? +0.5 * pow(2.0, (20.0 * t) - 10.0)
            : -0.5 * pow(2.0, 10.0 - (t * 20.0)) + 1.0;
}
#endif