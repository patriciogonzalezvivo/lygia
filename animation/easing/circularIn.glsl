/*
contributors: Hugh Kennedy (https://github.com/hughsk)
description: Circular in easing. From https://github.com/stackgl/glsl-easings
use: circularIn(<float> x)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/animation_easing.frag
*/

#ifndef FNC_CIRCULARIN
#define FNC_CIRCULARIN
float circularIn(in float t) { return 1.0 - sqrt(1.0 - t * t); }
#endif