/*
contributors: Ivan Dianov
description: polar to cartesian conversion.
use: polar2cart(<vec2> polar)
*/

#ifndef FNC_POLAR2CART
#define FNC_POLAR2CART
vec2 polar2cart(in vec2 polar) { return vec2(cos(polar.x), sin(polar.x)) * polar.y; }
#endif