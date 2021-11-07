#include "map.glsl"

/*
author:  Inigo Quiles
description: cast a ray
use: <float> castRay( in <vec3> pos, in <vec3> nor ) 
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

#ifndef RAYMARCH_SAMPLES
#define RAYMARCH_SAMPLES 64
#endif

#ifndef FNC_RAYMARCHCAST
#define FNC_RAYMARCHCAST

vec4 raymarchCast( in vec3 ro, in vec3 rd ) {
    float tmin = 1.0;
    float tmax = 20.0;
   
// #if defined(RAYMARCH_FLOOR)
//     float tp1 = (0.0-ro.y)/rd.y; if( tp1>0.0 ) tmax = min( tmax, tp1 );
//     float tp2 = (1.6-ro.y)/rd.y; if( tp2>0.0 ) { if( ro.y>1.6 ) tmin = max( tmin, tp2 );
//                                                  else           tmax = min( tmax, tp2 ); }
// #endif
    
    float t = tmin;
    vec3 m = vec3(-1.0);
    for ( int i = 0; i < RAYMARCH_SAMPLES; i++ ) {
        float precis = 0.0004*t;
        vec4 res = raymarchMap( ro + rd * t );
        if ( res.a < precis || t > tmax ) break;
        t += res.a;
        m = res.rgb;
    }

    #if defined(RAYMARCH_BACKGROUND) || defined(RAYMARCH_FLOOR)
    if ( t>tmax ) m = vec3(-1.0);
    #endif

    return vec4( m, t );
}

#endif