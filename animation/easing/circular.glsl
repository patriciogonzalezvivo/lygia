/*
original_author: Hugh Kennedy (https://github.com/hughsk)
description: circular easing. From https://github.com/stackgl/glsl-easings
use: circular<In|Out|InOut>(<float> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

#ifndef FNC_CIRCULARIN
#define FNC_CIRCULARIN
float circularIn(in float t) {
    return 1.0 - sqrt(1.0 - t * t);
}
#endif

#ifndef FNC_CIRCULAROUT
#define FNC_CIRCULAROUT
float circularOut(in float t) {
    return sqrt((2.0 - t) * t);
}
#endif

#ifndef FNC_CIRCULARINOUT
#define FNC_CIRCULARINOUT
float circularInOut(in float t) {
    return t < 0.5
        ? 0.5 * (1.0 - sqrt(1.0 - 4.0 * t * t))
        : 0.5 * (sqrt((3.0 - 2.0 * t) * (2.0 * t - 1.0)) + 1.0);
}
#endif
