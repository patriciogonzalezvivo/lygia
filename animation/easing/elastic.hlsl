#include "../../math/const.hlsl"

/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Elastic easing. From https://github.com/stackgl/glsl-easings
use: elastic<In|Out|InOut>(<float> x)
*/

#ifndef FNC_ELASTICIN
#define FNC_ELASTICIN
float elasticIn(in float t) {
    return sin(13.0 * t * HALF_PI) * pow(2.0, 10.0 * (t - 1.0));
}
#endif 

#ifndef FNC_ELASTICOUT
#define FNC_ELASTICOUT
float elasticOut(in float t) {
    return sin(-13.0 * (t + 1.0) * HALF_PI) * pow(2.0, -10.0 * t) + 1.0;
}
#endif

#ifndef FNC_ELASTICINOUT
#define FNC_ELASTICINOUT
float elasticInOut(in float t) {
    return t < 0.5
        ? 0.5 * sin(+13.0 * HALF_PI * 2.0 * t) * pow(2.0, 10.0 * (2.0 * t - 1.0))
        : 0.5 * sin(-13.0 * HALF_PI * ((2.0 * t - 1.0) + 1.0)) * pow(2.0, -10.0 * (2.0 * t - 1.0)) + 1.0;
}
#endif
