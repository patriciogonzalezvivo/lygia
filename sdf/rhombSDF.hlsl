#include "triSDF.hlsl"

/*
description: Returns a rhomb-shaped sdf
use: rhombSDF(<vec2> st)
author: Patricio Gonzalez Vivo
license: |
    Copyright (c) 2017 Patricio Gonzalez Vivo. All rights reserved.
    Distributed under BSD 3-clause "New" or "Revised" License. See LICENSE file at
    https://github.com/patriciogonzalezvivo/PixelSpiritDeck
*/

#ifndef FNC_RHOMBSDF
#define FNC_RHOMBSDF
float rhombSDF(float2 st) {
    return max(triSDF(st),
               triSDF(float2(st.x, 1. - st.y)));
}
#endif
