/*
original_author: Hugh Kennedy (https://github.com/hughsk)
description: quadrtic easing. From https://github.com/stackgl/glsl-easings
use: quadratic<In|Out|InOut>(<float> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

#ifndef FNC_QUADRATICIN
#define FNC_QUADRATICIN
float quadraticIn(in float t) {
    return t * t;
}
#endif

#ifndef FNC_QUADRATICOUT
#define FNC_QUADRATICOUT
float quadraticOut(in float t) {
    return -t * (t - 2.0);
}
#endif

#ifndef FNC_QUADRATICINOUT
#define FNC_QUADRATICINOUT
float quadraticInOut(in float t) {
    float p = 2.0 * t * t;
    return t < 0.5 ? p : -p + (4.0 * t) - 1.0;
}
#endif
