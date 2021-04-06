/*
author: Patricio Gonzalez Vivo
description: Returns a spiral SDF
use: spiralSDF(<vec2> st, <float> turns)
license: |
  Copyright (c) 2017 Patricio Gonzalez Vivo. All rights reserved.
  Distributed under BSD 3-clause "New" or "Revised" License. See LICENSE file at
  https://github.com/patriciogonzalezvivo/PixelSpiritDeck
*/

#ifndef FNC_SPIRALSDF
#define FNC_SPIRALSDF
float spiralSDF(vec2 st, float t) {
    st -= 0.5;
    float r = dot(st, st);
    float a = atan(st.y, st.x);
    return abs(sin(fract(log(r) * t + a * 0.159)));
}
#endif