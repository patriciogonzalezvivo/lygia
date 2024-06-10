#include "triangle.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: Returns the area of a triangle
use: <float3> normal(<Triangle> tri)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef FNC_TRIANGLE_AREA
#define FNC_TRIANGLE_AREA
float area(Triangle _tri) { return length( cross( _tri.b - _tri.a, _tri.c - _tri.a) ) * 0.5; }
#endif