/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Circular in/out easing. From https://github.com/stackgl/glsl-easings
use: circularInOut(<float> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

#ifndef FNC_CIRCULARINOUT
#define FNC_CIRCULARINOUT
float circularInOut(in float t) {
    return t < 0.5
        ? 0.5 * (1.0 - sqrt(1.0 - 4.0 * t * t))
        : 0.5 * (sqrt((3.0 - 2.0 * t) * (2.0 * t - 1.0)) + 1.0);
}
#endif
