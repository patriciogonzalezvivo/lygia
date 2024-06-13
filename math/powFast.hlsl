/*
contributors: Patricio Gonzalez Vivo
description: fast approximation to pow()
use: <float> powFast(<float> x, <float> exp)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_POWFAST
#define FNC_POWFAST

float powFast(const in float a, const in float b) { return a / ((1. - b) * a + b); }

#endif