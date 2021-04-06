/*
author: Patricio Gonzalez Vivo
description: fill a SDF. From PixelSpiritDeck https://github.com/patriciogonzalezvivo/PixelSpiritDeck
use: fill(<float> sdf, <float> size [, <float> edge])
license: |
  Copyright (c) 2017 Patricio Gonzalez Vivo. All rights reserved.
  Distributed under BSD 3-clause "New" or "Revised" License. See LICENSE file at
  https://github.com/patriciogonzalezvivo/PixelSpiritDeck
*/

#ifndef FNC_FILL
#define FNC_FILL

#include "aastep.glsl"

float fill(float x, float size, float edge) {
    return 1.0 - smoothstep(size - edge, size + edge, x);
}

float fill(float x, float size) {
    return 1.0 - aastep(size, x);
}
#endif
