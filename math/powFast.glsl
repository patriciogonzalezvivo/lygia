/*
contributors: Patricio Gonzalez Vivo
description: fast approximation to pow()
use: <float> powFast(<float> x, <float> exp)
*/

#ifndef FNC_POWFAST
#define FNC_POWFAST

float powFast(const in float a, const in float b) { return a / ((1. - b) * a + b); }

#endif