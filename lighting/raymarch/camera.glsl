/*
author:  Inigo Quiles
description: set a camera for raymarching 
use: <mat3> raymarchCamera(in <vec3> ro, [in <vec3> ta [, in <vec3> up] ])
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

#ifndef FNC_RAYMARCHCAMERA
#define FNC_RAYMARCHCAMERA

mat3 raymarchCamera( in vec3 ro, in vec3 ta, in vec3 up ) {
    vec3 cw = normalize(ta-ro);
    vec3 cp = up;
    vec3 cu = normalize( cross(cw,cp) );
    vec3 cv = normalize( cross(cu,cw) );
    return mat3( cu, cv, cw );
}

mat3 raymarchCamera( in vec3 ro, in vec3 ta, float cr ) {
    vec3 cw = normalize(ta-ro);
    vec3 cp = vec3(sin(cr), cos(cr),0.0);
    vec3 cu = normalize( cross(cw,cp) );
    vec3 cv =          ( cross(cu,cw) );
    return mat3( cu, cv, cw );
}

mat3 raymarchCamera( in vec3 ro, in vec3 ta ) {
    return raymarchCamera( ro, ta, vec3(0.0, 1.0, 0.0) );
}

mat3 raymarchCamera( in vec3 ro ) {
    return raymarchCamera( ro, vec3(0.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0) );
}

#endif