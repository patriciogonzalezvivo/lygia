#include ../math/spherical.glsl
#include ../math/superFormula.glsl

/*
author: Kathy McGuiness
description: Returns 2D supershape
use: supershapeSDF(in <float> theta, in <float> size s, in <float> M)
*/

#ifndef FNC_SUPERSHAPESDF
#define FNC_SUPERSHAPESDF
float supershapeSDF( in vec2 st, in float s, in float m ) {
  vec2 q;
  float d = spherical( st ).x;
  float a = spherical( st ).y;
  float r = superFormula(a, 1.0, 1.0, 0.3, 0.3, 0.3, m);
  q.x = s * r * cos(a);
  q.y = s * r * sin(a);
  return d -= length(q); 
}
#endif