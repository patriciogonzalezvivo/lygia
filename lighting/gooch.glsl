#include "material/roughness.glsl"
#include "material/normal.glsl"
#include "material/baseColor.glsl"

#include "diffuse.glsl"
#include "specular.glsl"

#include "../sample/textureShadowPCF.glsl"
#include "../color/space/linear2gamma.glsl"

/*
author: Patricio Gonzalez Vivo
description: render with a gooch stylistic shading model
use: <vec4> gooch(<vec4> baseColor, <vec3> normal, <vec3> light, <vec3> view, <float> roughness)
options:
    - GOOCH_WARM: defualt vec3(0.25, 0.15, 0.0)
    - GOOCH_COLD: defualt vec3(0.0, 0.0, 0.2)
    - GOOCH_SPECULAR: defualt vec3(1.0, 1.0, 1.0)
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - LIGHT_COORD:       in GlslViewer is  v_lightCoord
    - LIGHT_SHADOWMAP:   in GlslViewer is u_lightShadowMap
    - LIGHT_SHADOWMAP_SIZE: in GlslViewer is 1024.0

license: |
    Copyright (c) 2021 Patricio Gonzalez Vivo.
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef GOOCH_WARM 
#define GOOCH_WARM vec3(0.25, 0.15, 0.0)
#endif 

#ifndef GOOCH_COLD 
#define GOOCH_COLD vec3(0.0, 0.0, 0.2)
#endif 

#ifndef GOOCH_SPECULAR
#define GOOCH_SPECULAR vec3(1.0, 1.0, 1.0)
#endif 

#ifndef FNC_GOOCH
#define FNC_GOOCH
vec4 gooch(vec4 baseColor, vec3 normal, vec3 light, vec3 view, float roughness) {
    vec3 warm = GOOCH_WARM + baseColor.rgb * 0.6;
    vec3 cold = GOOCH_COLD + baseColor.rgb * 0.1;

    vec3 l = normalize(light);
    vec3 n = normalize(normal);
    vec3 v = normalize(view);

    // Lambert Diffuse
    float diffuse = diffuse(l, n, v, roughness);
    // Phong Specular
    float specular = specular(l, n, v, roughness);

#if defined(LIGHT_SHADOWMAP) && defined(LIGHT_SHADOWMAP_SIZE) && defined(LIGHT_COORD) && !defined(PLATFORM_RPI) && !defined(PLATFORM_WEBGL)
    float bias = 0.005;
    float shadow = textureShadowPCF(u_lightShadowMap, vec2(LIGHT_SHADOWMAP_SIZE), (LIGHT_COORD).xy, (LIGHT_COORD).z - bias);
    specular *= shadow;
    diffuse *= shadow;
#endif

    return linear2gamma( vec4(mix(mix(cold, warm, diffuse), GOOCH_SPECULAR, specular), baseColor.a) );
}

#endif