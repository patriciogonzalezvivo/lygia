/*
original_author: Daniel Shiffman
description: Returns r
use: superFormula(in <float> theta, in <float> a, in <float> b, in <float> n1, in <float> n2, in <float> n3, in <float> m)
a, b, n1, n2, n3, m parameters
*/

#ifndef FNC_SUPERFORMULA
#define FNC_SUPERFORMULA
float superFormula(in float theta, in float a, in float b, in float n1, in float n2, in float n3, in float m) {
  float t1 = abs((1.0/a) * cos(m * theta / 4.0));
  t1 = pow(t1, n2);
  
  float t2 = abs((1.0/b) * sin(m * theta / 4.0));
  t2 = pow(t2, n3);
  
  float t3 = t1 + t2;
  float r = pow(t3, -1.0 / n1);
  return r;
}
#endif