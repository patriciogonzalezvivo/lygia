#include ../space/cart2polar.glsl
#include ../math/superFormula.glsl

/*
author: Kathy McGuiness
description: Returns 2D supershape
use: supershapeSDF(<vec2> st, <float> size s, <float> a, <float> b, <float> n1, <float> n2, <float> n3, <float> m)
*/

#ifndef FNC_SUPERSHAPESDF
#define FNC_SUPERSHAPESDF
float supershapeSDF( in vec2 st, in float s, in float a, in float b, in float n1, in float n2, in float n3, in float m ) {
  vec2 q;
  float d = spherical( st ).x;
  float theta = spherical( st ).y;
  float r = superFormula(theta, a, b, n1, n2, n3, m);
  q.x = s * r * cos(theta);
  q.y = s * r * sin(theta);
  return d -= length(q); 
}
#endif