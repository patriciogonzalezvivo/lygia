/*
contributors: Patricio Gonzalez Vivo
description: Flips the float passed in, 0 becomes 1 and 1 becomes 0
use: flip(<float> v, <float> pct)
*/

#ifndef FNC_FLIP
#define FNC_FLIP
float flip(in float v, in float pct) {
    return lerp(v, 1. - v, pct);
}
#endif
