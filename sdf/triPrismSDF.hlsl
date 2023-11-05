/*
contributors:  Inigo Quiles
description: generate the SDF of a triangular prism
use: <float> triPrismSDF( in <float3> pos, in <float2> h ) 
license: |
    The MIT License
    Copyright © 2013 Inigo Quilez
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    https://www.youtube.com/c/InigoQuilez
    https://iquilezles.org/

    A list of useful distance function to simple primitives. All
    these functions (except for ellipsoid) return an exact
    euclidean distance, meaning they produce a better SDF than
    what you'd get if you were constructing them from boolean
    operations (such as cutting an infinite cylinder with two planes).

    List of other 3D SDFs:
       https://www.shadertoy.com/playlist/43cXRl
    and
       http://iquilezles.org/www/articles/distfunctions/distfunctions.htm
*/
#ifndef FNC_TRIPRISMSDF
#define FNC_TRIPRISMSDF

float triPrismSDF( float3 p, float2 h ) {
    float3 q = abs(p);
    return max(q.z-h.y,max(q.x*0.866025+p.y*0.5,-p.y)-h.x*0.5);
}

// float triPrismSDF( float3 p, float2 h ) {
//     const float k = sqrt(3.0);
//     h.x *= 0.5*k;
//     p.xy /= h.x;
//     p.x = abs(p.x) - 1.0;
//     p.y = p.y + 1.0/k;
//     if( p.x+k*p.y>0.0 ) p.xy=float2(p.x-k*p.y,-k*p.x-p.y)/2.0;
//     p.x -= clamp( p.x, -2.0, 0.0 );
//     float d1 = length(p.xy)*sign(-p.y)*h.x;
//     float d2 = abs(p.z)-h.y;
//     return length(max(float2(d1,d2),0.0)) + min(max(d1,d2), 0.);
// }


#endif