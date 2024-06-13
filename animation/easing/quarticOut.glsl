/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Quartic out easing. From https://github.com/stackgl/glsl-easings
use: quarticOut<float> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

#ifndef FNC_QUARTICOUT
#define FNC_QUARTICOUT
float quarticOut(in float t) { 
    float it = t - 1.0;
    return it * it * it * (1.0 - t) + 1.0; 
}
#endif