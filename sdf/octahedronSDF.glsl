/*
author:  Inigo Quiles
description: generate the SDF of a octahedron
use: <float> octahedronSDF(<vec3> p, <float> s)
options:
    OCTAHEDRON_EXACT_DISTANCE
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



#ifndef FNC_OCTAHEDRONSDF
#define FNC_OCTAHEDRONSDF

float octahedronSDF(vec3 p, float s) {
    p = abs(p);
    float m = p.x + p.y + p.z - s;

#ifdef OCTAHEDRON_EXACT_DISTANCE
    // exact distance
    vec3 o = min(3.0*p - m, 0.0);
    o = max(6.0*p - m*2.0 - o*3.0 + (o.x+o.y+o.z), 0.0);
    return length(p - s*o/(o.x+o.y+o.z));

// #elif OCTAHEDRON_EXACT_DISTANCE == 2
//     // exact distance
//     vec3 q = vec3(0.0);
//          if( 3.0*p.x < m ) q = p.xyz;
//     else if( 3.0*p.y < m ) q = p.yzx;
//     else if( 3.0*p.z < m ) q = p.zxy;
//     else return m*0.57735027;
//     float k = clamp(0.5*(q.z-q.y+s),0.0,s); 
//     return length(vec3(q.x,q.y-s+k,q.z-k)); 
    
#else
    // bound, not exact
    return m*0.57735027;
#endif
}

#endif