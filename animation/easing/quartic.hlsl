/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Quartic easing. From https://github.com/stackgl/glsl-easings
use: quartic<In|Out|InOut>(<float> x)
*/

#ifndef FNC_QUARTICIN
#define FNC_QUARTICIN
float quarticIn(in float t) {
    return pow(t, 4.0);
}
#endif

#ifndef FNC_QUARTICOUT
#define FNC_QUARTICOUT
float quarticOut(in float t) {
    return pow(t - 1.0, 3.0) * (1.0 - t) + 1.0;
}
#endif

#ifndef FNC_QUARTICINOUT
#define FNC_QUARTICINOUT
float quarticInOut(in float t) {
    return t < 0.5
      ? +8.0 * pow(t, 4.0)
      : -8.0 * pow(t - 1.0, 4.0) + 1.0;
}
#endif
