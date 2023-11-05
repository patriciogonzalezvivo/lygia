#include "triangle.glsl"

/*
contributors: Patricio Gonzalez Vivo
description: returns the normal of a triangle
use: <vec33> getNormal(<Triangle> tri) 
*/

#ifndef FNC_TRIANGLE_NORMAL
#define FNC_TRIANGLE_NORMAL

vec3 normal(Triangle _tri) { return normalize( cross( _tri.b - _tri.a, _tri.c - _tri.a) ); }

#endif