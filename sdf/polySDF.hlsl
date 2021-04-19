#include "../math/const.hlsl"

/*
author: Patricio Gonzalez Vivo
description: Returns a sdf for a regular polygon with V sides.
use: polySDF(<float2> st, int V)
license: |
  Copyright (c) 2017 Patricio Gonzalez Vivo. All rights reserved.
  Distributed under BSD 3-clause "New" or "Revised" License. See LICENSE file at
  https://github.com/patriciogonzalezvivo/PixelSpiritDeck
*/

#ifndef FNC_POLYSDF
#define FNC_POLYSDF
float polySDF(in float2 st, in int V) {
    st = st * 2. - 1.;
    float a = atan2(st.x, st.y) + PI;
    float r = length(st);
    float v = TAU / float(V);
    return cos(floor(.5 + a / v) * v - a ) * r;
}
#endif
