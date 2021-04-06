/*
author: Patricio Gonzalez Vivo
description: Flips the float passed in, 0 becomes 1 and 1 becomes 0
use: flip(<float> v, <float> pct)
license: |
  Copyright (c) 2017 Patricio Gonzalez Vivo. All rights reserved.
  Distributed under BSD 3-clause "New" or "Revised" License. See LICENSE file at
  https://github.com/patriciogonzalezvivo/PixelSpiritDeck
*/

#ifndef FNC_FLIP
#define FNC_FLIP
float flip(in float v, in float pct) {
    return mix(v, 1. - v, pct);
}
#endif
