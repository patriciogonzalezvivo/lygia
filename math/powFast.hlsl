/*
original_author: Patricio Gonzalez Vivo
description: fast approximation to pow()
use: powFast(<float> x, <float> exp)
*/

#ifndef FNC_POWFAST
#define FNC_POWFAST

float powFast(float a, float b) {
  return a / ((1. - b) * a + b);
}

#endif