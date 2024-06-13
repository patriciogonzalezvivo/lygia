#include "../../math/const.hlsl"

/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Sine easing. From https://github.com/stackgl/glsl-easings
use: sine<In|Out|InOut>(<float> x)
*/

#ifndef FNC_SINEIN
#define FNC_SINEIN
float sineIn(in float t) {
    return sin((t - 1.0) * HALF_PI) + 1.0;
}
#endif

#ifndef FNC_SINEOUT
#define FNC_SINEOUT
float sineOut(in float t) {
    return sin(t * HALF_PI);
}
#endif

#ifndef FNC_SINEINOUT
#define FNC_SINEINOUT
float sineInOut(in float t) {
    return -0.5 * (cos(PI * t) - 1.0);
}
#endif
