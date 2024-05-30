/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Quintic easing. From https://github.com/stackgl/glsl-easings
use: quintic<In|Out|InOut>(<float> x)
*/

#ifndef FNC_QUINTICIN
#define FNC_QUINTICIN
float quinticIn(in float t) {
    return pow(t, 5.0);
}
#endif

#ifndef FNC_QUINTICOUT
#define FNC_QUINTICOUT
float quinticOut(in float t) {
    return 1.0 - (pow(t - 1.0, 5.0));
}
#endif

#ifndef FNC_QUINTICINOUT
#define FNC_QUINTICINOUT
float quinticInOut(in float t) {
    return t < 0.5
        ? +16.0 * pow(t, 5.0)
        : -0.5 * pow(2.0 * t - 2.0, 5.0) + 1.0;
}
#endif
