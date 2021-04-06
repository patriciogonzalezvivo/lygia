/*
author: Patricio Gonzalez Vivo
description: fill a stroke in a SDF. From PixelSpiritDeck https://github.com/patriciogonzalezvivo/PixelSpiritDeck
use: stroke(<float> sdf, <float> size, <float> width [, <float> edge])
license: |
  Copyright (c) 2017 Patricio Gonzalez Vivo. All rights reserved.
  Distributed under BSD 3-clause "New" or "Revised" License. See LICENSE file at
  https://github.com/patriciogonzalezvivo/PixelSpiritDeck
*/

#ifndef FNC_STROKE
#define FNC_STROKE

#include "aastep.glsl"
#include "../math/saturate.glsl"

float stroke(float x, float size, float w) {
    float d = aastep(size, x + w * 0.5) - aastep(size, x - w * 0.5);
    return saturate(d);
}

float stroke(float x, float size, float w, float edge) {
    float d = smoothstep(size - edge, size + edge, x + w * 0.5) - smoothstep(size - edge, size + edge, x - w * 0.5);
    return saturate(d);
}

#endif
