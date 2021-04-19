#include "circleSDF.hlsl"

/*
author: Patricio Gonzalez Vivo
description: Returns an almond-shaped sdf
use: vesicaSDF(<float2> st, <float> w)
license: |
  Copyright (c) 2017 Patricio Gonzalez Vivo. All rights reserved.
  Distributed under BSD 3-clause "New" or "Revised" License. See LICENSE file at
  https://github.com/patriciogonzalezvivo/PixelSpiritDeck
*/

#ifndef FNC_VESICASDF
#define FNC_VESICASDF
float vesicaSDF(in float2 st, in float w) {
    float2 offset = float2(w*.5,0.);
    return max( circleSDF(st-offset),
                circleSDF(st+offset));
}
#endif
