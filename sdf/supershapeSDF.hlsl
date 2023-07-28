#include ../space/cart2polar.hlsl
#include ../math/superFormula.hlsl

/*
author: Kathy McGuiness
supershape_author: Paul Bourke
description: Returns 2D supershape
use: supershapeSDF(<vec2> st, <float> size s, <float> a, <float> b, <float> n1, <float> n2, <float> n3, <float> m)
Notes about parameters:
*m determines number of sides/branches; m = 0 yields a circle
*a!=b results in an assymetrical shape
*n1=n2=n3<1, the shape is "pinched"
*n1>n2,n3, the shape is "bloated"
*n1!=n2!=n3, the shape is assymetrical
*/

#ifndef FNC_SUPERSHAPESDF
#define FNC_SUPERSHAPESDF
float supershapeSDF( in vec2 st, in float s, in float a, in float b, in float n1, in float n2, in float n3, in float m ) {
  vec2 q;
  float d = cart2polar( st ).y;
  float theta = cart2polar( st ).x;
  float r = superFormula(theta, a, b, n1, n2, n3, m);
  q.x = s * r * cos(theta);
  q.y = s * r * sin(theta);
  return d -= length(q); 
}
#endif