#include "../math/const.glsl"

/*
author: Patricio Gonzalez Vivo
description: Returns a sdf for rays with N branches
use: raysSDF(<vec2> st, <int> N)
license: |
  Copyright (c) 2017 Patricio Gonzalez Vivo. All rights reserved.
  Distributed under BSD 3-clause "New" or "Revised" License. See LICENSE file at
  https://github.com/patriciogonzalezvivo/PixelSpiritDeck
*/

#ifndef FNC_RAYSSDF
#define FNC_RAYSSDF
float raysSDF(in vec2 st, in int N) {
    st -= .5;
    return fract(atan(st.y, st.x) / TAU * float(N));
}
#endif
