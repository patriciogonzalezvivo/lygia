/*
original_author: Patricio Gonzalez Vivo
description: bias high pass
use: highPass(<float> value, <float> bias)
*/

#ifndef FNC_HIGHPASS
#define FNC_HIGHPASS

float highPass(in float value, in float bias) { return max(value - bias, 0.0) / (1.0 - bias); }

#endif
