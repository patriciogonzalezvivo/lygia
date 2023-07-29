#include "../space/cart2polar.glsl"
#include "../math/superFormula.glsl"

/*
original_author: Kathy McGuiness
description: |
    It returns a 3D supershape--a 3d extension of the supershape, which is a mathematical function for modelling natural forms develop by [Paul Bourke](http://paulbourke.net/geometry/) and Johan Gielis. 
    Some notes about the parameters:
        * `a=b=1.0` yields best shapes
        * `m` determines number of sides
        * `m = 0` or `n2=n3` yields a sphere
        * `n1!=n2!=n3` yields an assymetrical 3d shape
        * `n2=n3<2` the shape more spherical

    For more information about the 3D supershape, check [this website](https://www.syedrezaali.com/3d-supershapes/).
    
use: <float> supershape3dSDF(<float3> st, <float> size s, <float> a, <float> b, <float> n1, <float> n2, <float> n3, <float> m)
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_supershape.frag
*/

#ifndef FNC_SUPERSHAPE3DSDF
#define FNC_SUPERSHAPE3DSDF
float superShape3dSDF( in float3 st, float s, float a, float b, float n1, float n2, float n3, float m) {
  float r = cart2polar( st ).x;
  float theta = cart2polar( st ).y;
  float phi = cart2polar( st ).z;
  float r1 = superFormula( phi, a, b, n1, n2, n3, m );
  float r2 = superFormula( theta, a, b, n1, n2, n3, m );
  float d1 = s * r1 * cos( phi ) * r2 * cos( theta );
  float d2 = s * r1 * sin( phi ) * r2 * cos( theta );
  float d3 = s * r2 * sin( theta ) ;
  float3 q = float3(d1, d2, d3);
  return r -= length(q); 
 }
#endif