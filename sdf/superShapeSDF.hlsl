#include ../space/cart2polar.hlsl
#include ../math/superFormula.hlsl

/*
original_author: Kathy McGuiness
description: |
    It returns a supershape, which is a mathematical function for modelling natural forms develop by [Paul Bourke](http://paulbourke.net/geometry/) and Johan Gielis. 
    Some nots about the parameters:

        * `m` determines number of sides/branches
        * `m = 0` yields a circle
        * `a!=b` results in an assymetrical shape
        * `n1=n2=n3<1` the shape is "pinched"
        * `n1>n2,n3` the shape is "bloated"
        * `n1!=n2!=n3` the shape is assymetrical

use: <float> supershapeSDF(<float2> st, <float> size s, <float> a, <float> b, <float> n1, <float> n2, <float> n3, <float> m)
*/

#ifndef FNC_SUPERSHAPESDF
#define FNC_SUPERSHAPESDF
float superShapeSDF( in float2 st, in float s, in float a, in float b, in float n1, in float n2, in float n3, in float m ) {
    float2 polar = cart2polar( st );
    float d = polar.y;
    float theta = polar.x;
    float r = superFormula(theta, a, b, n1, n2, n3, m);
    float2 q = s * r * float2(cos(theta), sin(theta));
    return d - length(q); 
}
#endif