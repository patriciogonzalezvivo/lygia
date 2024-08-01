#include "triangle.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Returns the centroid of a triangle
use: <vec3> centroid(<Triangle> tri)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_TRIANGLE_CENTROID
#define FNC_TRIANGLE_CENTROID

vec3 centroid(Triangle _tri) { return (_tri.a + _tri.b + _tri.c) * 0.3333333333333; }

#endif