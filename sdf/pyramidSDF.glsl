/*
author:  Inigo Quiles
description: generate the SDF of s pyramid
use: <float> pyramidSDF(<vec3> p, <float> h )
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
#ifndef FNC_PYRAMIDSDF
#define FNC_PYRAMIDSDF

float pyramidSDF( in vec3 p, in float h )  {
   float m2 = h*h + 0.25;
   
   // symmetry
   p.xz = abs(p.xz);
   p.xz = (p.z>p.x) ? p.zx : p.xz;
   p.xz -= 0.5;

   // project into face plane (2D)
   vec3 q = vec3( p.z, h*p.y - 0.5*p.x, h*p.x + 0.5*p.y);

   float s = max(-q.x,0.0);
   float t = clamp( (q.y-0.5*p.z)/(m2+0.25), 0.0, 1.0 );
   
   float a = m2*(q.x+s)*(q.x+s) + q.y*q.y;
   float b = m2*(q.x+0.5*t)*(q.x+0.5*t) + (q.y-m2*t)*(q.y-m2*t);
   
   float d2 = min(q.y,-q.x*m2-q.y*0.5) > 0.0 ? 0.0 : min(a,b);
   
   // recover 3D and scale, and add sign
   return sqrt( (d2+q.z*q.z)/m2 ) * sign(max(q.z,-p.y));;
}


#endif