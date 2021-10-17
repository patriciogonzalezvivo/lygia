#include "map.glsl"

/*
author:  Inigo Quiles
description: calculate normals http://iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
use: <float> raymarchNormal( in <vec3> pos ) 
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


#ifndef FNC_RAYMARCHNORMAL
#define FNC_RAYMARCHNORMAL

vec3 raymarchNormal( in vec3 pos ) {
    vec2 e = vec2(1.0, -1.0) * 0.5773 * 0.0005;
    return normalize( e.xyy * raymarchMap( pos + e.xyy ).a + 
                      e.yyx * raymarchMap( pos + e.yyx ).a + 
                      e.yxy * raymarchMap( pos + e.yxy ).a + 
                      e.xxx * raymarchMap( pos + e.xxx ).a );
}

#endif