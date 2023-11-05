/*
contributors: Patricio Gonzalez Vivo
description: heatmap palette
use: <vec3> heatmap(<float> value)
examples:
    - https://raw.githubusercontent.com/eduardfossas/lygia-study-examples/main/color/palette/heatmap.frag
*/

#ifndef FNC_HEATMAP
#define FNC_HEATMAP
vec3 heatmap(float v) {
    vec3 r = v * 2.1 - vec3(1.8, 1.14, 0.3);
    return 1.0 - r * r;
}
#endif