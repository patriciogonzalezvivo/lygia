#include "../lighting/raymarch/camera.glsl"

/*
author: Patricio Gonzalez Vivo
description: Displace UV space into a XYZ space using an heightmap
use: <vec3> displace(<sampler2D> tex, <vec3> ro, <vec3|vec2> rd) 
license: |
  Copyright (c) 2022 Patricio Gonzalez Vivo.
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    

*/

#ifndef DISPLACE_DEPTH
#define DISPLACE_DEPTH 1.
#endif

#ifndef DISPLACE_PRECISION
#define DISPLACE_PRECISION 0.01
#endif

#ifndef DISPLACE_SAMPLER
#define DISPLACE_SAMPLER(UV) texture2D(tex, UV).r
#endif

#ifndef DISPLACE_MAX_ITERATIONS
#define DISPLACE_MAX_ITERATIONS 120
#endif

#ifndef FNC_DISPLACE
#define FNC_DISPLACE
vec3 displace(sampler2D tex, vec3 ro, vec3 rd) {

    // the z length of the target vector
    float dz = ro.z - DISPLACE_DEPTH;
    float t = dz / rd.z;

    // the intersection point between the ray and the hightest point on the plane
    vec3 prev = vec3(
        ro.x - rd.x * t,
        ro.y - rd.y * t,
        ro.z - rd.z * t
    );
    
    vec3 curr = prev;
    float lastD = prev.z;
    float hmap = 0.;
    float df = 0.;
    
    for (int i = 0; i < DISPLACE_MAX_ITERATIONS; i++) {
        prev = curr;
        curr = prev + rd * DISPLACE_PRECISION;

        hmap = DISPLACE_SAMPLER( curr.xy - 0.5 );
        // distance to the displaced surface
        float df = curr.z - hmap * DISPLACE_DEPTH;
        
        // if we have an intersection
        if (df < 0.0) {
            // linear interpolation to find more precise df
            float t = lastD / (abs(df)+lastD);
            return (prev + t * (curr - prev)) + vec3(0.5, 0.5, 0.0);
        } 
        else
            lastD = df;
    }
    
    return vec3(0.0, 0.0, 1.0);
}

vec3 displace(sampler2D tex, vec3 ro, vec2 uv) {
    vec3 rd = raymarchCamera(ro) * normalize(vec3(uv - 0.5, 1.0));
    return displace(u_tex0Depth, ro, rd);
}
#endif