#include "../math/aastep.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: fill a SDF. From PixelSpiritDeck https://github.com/patriciogonzalezvivo/PixelSpiritDeck
use: fill(<float> sdf, <float> size [, <float> edge])
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
*/

#ifndef FNC_FILL
#define FNC_FILL
float fill(float x, float size, float edge) {
    return 1.0 - smoothstep(size - edge, size + edge, x);
}

float fill(float x, float size) {
    return 1.0 - aastep(size, x);
}
#endif
