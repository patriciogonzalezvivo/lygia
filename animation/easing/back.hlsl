#include "../../math/const.hlsl"

/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Back easing. From https://github.com/stackgl/glsl-easings
use: back<In|Out|InOut>(<float> x)
*/

#ifndef FNC_BACKIN
#define FNC_BACKIN
float backIn(in float t) {
    return pow(t, 3.) - t * sin(t * PI);
}
#endif

#ifndef FNC_BACKOUT
#define FNC_BACKOUT
float backOut(in float t) {
    return 1. - backIn(1. - t);
}
#endif

#ifndef FNC_BACKINOUT
#define FNC_BACKINOUT
float backInOut(in float t) {
    float f = t < .5
        ? 2.0 * t
        : 1.0 - (2.0 * t - 1.0);

    float g = backIn(f);

  return t < 0.5
      ? 0.5 * g
      : 0.5 * (1.0 - g) + 0.5;
}
#endif
