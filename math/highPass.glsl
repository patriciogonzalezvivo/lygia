/*
contributors: Patricio Gonzalez Vivo
description: bias high pass
use: <float> highPass(<float> value, <float> bias)
*/

#ifndef FNC_HIGHPASS
#define FNC_HIGHPASS
float highPass(in float v, in float b) { return max(v - b, 0.0) / (1.0 - b); }
#endif
