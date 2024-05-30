#include "../../math/const.hlsl"

/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Bounce easing. From https://github.com/stackgl/glsl-easings
use: bounce<In|Out|InOut>(<float> x)
*/

#ifndef FNC_BOUNCEOUT
#define FNC_BOUNCEOUT
float bounceOut(in float t) {
    const float a = 4.0 / 11.0;
    const float b = 8.0 / 11.0;
    const float c = 9.0 / 10.0;

    const float ca = 4356.0 / 361.0;
    const float cb = 35442.0 / 1805.0;
    const float cc = 16061.0 / 1805.0;

    float t2 = t * t;

    return t < a
        ? 7.5625 * t2
        : t < b
            ? 9.075 * t2 - 9.9 * t + 3.4
            : t < c
                ? ca * t2 - cb * t + cc
                : 10.8 * t * t - 20.52 * t + 10.72;
}
#endif

#ifndef FNC_BOUNCEIN
#define FNC_BOUNCEIN
float bounceIn(in float t) {
    return 1.0 - bounceOut(1.0 - t);
}
#endif

#ifndef FNC_BOUNCEINOUT
#define FNC_BOUNCEINOUT
float bounceInOut(in float t) {
    return t < 0.5
        ? 0.5 * (1.0 - bounceOut(1.0 - t * 2.0))
        : 0.5 * bounceOut(t * 2.0 - 1.0) + 0.5;
}
#endif
