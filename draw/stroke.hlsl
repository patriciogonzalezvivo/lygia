/*
contributors: Patricio Gonzalez Vivo
description: fill a stroke in a SDF. From PixelSpiritDeck https://github.com/patriciogonzalezvivo/PixelSpiritDeck
use: stroke(<float> sdf, <float> size, <float> width [, <float> edge])
*/

#ifndef FNC_STROKE
#define FNC_STROKE

#include "../math/aastep.hlsl"

float stroke(float x, float size, float w) {
    float d = aastep(size, x + w * 0.5) - aastep(size, x - w * 0.5);
    return saturate(d);
}

float stroke(float x, float size, float w, float edge) {
    float d = smoothstep(size - edge, size + edge, x + w * 0.5) - smoothstep(size - edge, size + edge, x - w * 0.5);
    return saturate(d);
}

#endif
