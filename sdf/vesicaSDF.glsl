#include "circleSDF.glsl"

/*
author: Patricio Gonzalez Vivo
description: Returns an almond-shaped sdf
use: vesicaSDF(<vec2> st, <float> w)
license: |
  Copyright (c) 2017 Patricio Gonzalez Vivo. All rights reserved.
  Distributed under BSD 3-clause "New" or "Revised" License. See LICENSE file at
  https://github.com/patriciogonzalezvivo/PixelSpiritDeck
*/

#ifndef FNC_VESICASDF
#define FNC_VESICASDF
float vesicaSDF(in vec2 st, in float w) {
    vec2 offset = vec2(w*.5,0.);
    return max( circleSDF(st-offset),
                circleSDF(st+offset));
}
#endif
