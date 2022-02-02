#include "../math/saturate.glsl"

/*
author: Patricio Gonzalez Vivo
description: Simple single pass fluid simlation from the book GPU Pro 2, "Simple and Fast Fluids"
use: <vec2> fluidSolver(<sampler2D> tex, <vec2> st, <vec2> pixel, <vec2> force)
options:
    - FLUIDSOLVER_DT
    - FLUIDSOLVER_VORTICITY
    - FLUIDSOLVER_VISCOSITY
    - FLUIDSOLVER_SAMPLER_FNC
    - FLUIDSOLVER_PACK_FNC
license: |
  Copyright (c) 2022 Patricio Gonzalez Vivo.
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef FLUIDSOLVER_DT
#define FLUIDSOLVER_DT 0.15
#endif

// lower value for vorticity threshold means higher viscosity
// and vice versa (max .3). Setting it to 0. disables it.
#ifndef FLUIDSOLVER_VORTICITY
#define FLUIDSOLVER_VORTICITY 0.0015
#endif


// higher this threshold, lower the viscosity (max .8)
#ifndef FLUIDSOLVER_VISCOSITY
#define FLUIDSOLVER_VISCOSITY .16
#endif

// #define FLUIDSOLVER_SAMPLER_FNC(UV) (texture2D(tex, UV)) 
#ifndef FLUIDSOLVER_SAMPLER_FNC
#define FLUIDSOLVER_SAMPLER_FNC(UV) (texture2D(tex, UV) * vec4(vec2(2.0), 3.0, 2.0) - vec4(vec2(1.0), 0.0, 1.0))
#endif

#ifndef FLUIDSOLVER_PACK_FNC
#define FLUIDSOLVER_PACK_FNC(COLOR) saturate(COLOR * vec4(vec2(0.5), 1./3., 0.5) + vec4(vec2(0.5), 0.0, 0.5))
#endif

#ifndef FNC_FLUIDSOLVER
#define FNC_FLUIDSOLVER

vec4 fluidSolver(sampler2D tex, vec2 st, vec2 pixel, vec2 force) {
    float k = .2;
    float s = k/FLUIDSOLVER_DT;
    
    vec4 fluidData = FLUIDSOLVER_SAMPLER_FNC(st);
    vec4 fr = FLUIDSOLVER_SAMPLER_FNC(st + vec2(pixel.x, 0.));
    vec4 fl = FLUIDSOLVER_SAMPLER_FNC(st - vec2(pixel.x, 0.));
    vec4 ft = FLUIDSOLVER_SAMPLER_FNC(st + vec2(0., pixel.y));
    vec4 fd = FLUIDSOLVER_SAMPLER_FNC(st - vec2(0., pixel.y));

    vec3 ddx = (fr - fl).xyz * 0.5;
    vec3 ddy = (ft - fd).xyz * 0.5;
    float divergence = ddx.x + ddy.y;
    vec2 densityDiff = vec2(ddx.z, ddy.z);
    
    // Solving for density
    fluidData.z -= FLUIDSOLVER_DT * dot(vec3(densityDiff, divergence), fluidData.xyz);
    
    // Solving for velocity
    vec2 laplacian = fr.xy + fl.xy + ft.xy + fd.xy - 4.*fluidData.xy;
    vec2 viscosityForce = FLUIDSOLVER_VISCOSITY * laplacian;
    
    // Semi-lagrangian advection
    vec2 densityInvariance = s * densityDiff;
    vec2 was = st - FLUIDSOLVER_DT * fluidData.xy * pixel;
    fluidData.xyw = FLUIDSOLVER_SAMPLER_FNC(was).xyw;
    fluidData.xy += FLUIDSOLVER_DT*(viscosityForce - densityInvariance + clamp(force, -1., 1.) * 10.0);
    
    // velocity decay
    fluidData.xy = max(vec2(0.), abs(fluidData.xy) - 5e-6)*sign(fluidData.xy);
    
    // Vorticity confinement
    fluidData.w = (fd.x - ft.x + fr.y - fl.y); // curl stored in the w channel
    vec2 vorticity = vec2(abs(ft.w) - abs(fd.w), abs(fl.w) - abs(fr.w));
    vorticity *= FLUIDSOLVER_VORTICITY/(length(vorticity) + 1e-5) * fluidData.w;
    fluidData.xy += vorticity;

    // Boundary conditions
    fluidData.x *= smoothstep(0.5, 0.49,abs(st.x - 0.5));
    fluidData.y *= smoothstep(0.5, 0.48,abs(st.y - 0.5));
    
    return FLUIDSOLVER_PACK_FNC(fluidData);
}

#endif