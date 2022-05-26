#include "../math/saturate.glsl"

/*
author: Patricio Gonzalez Vivo
description: Simple single pass fluid simlation from the book GPU Pro 2, "Simple and Fast Fluids"
use: <vec2> fluidSolver(<sampler2D> tex, <vec2> st, <vec2> pixel, <vec2> force)
options:
    - FLUIDSOLVER_DT: Default 0.15
    - FLUIDSOLVER_DX: Defailt 1.0
    - FLUIDSOLVER_VORTICITY: lower value for vorticity threshold means higher viscosity and vice versa (max .3)
    - FLUIDSOLVER_VISCOSITY: Default: 0.16
    - FLUIDSOLVER_VELOCITY_DECAY: Default: 5e-6
    - FLUIDSOLVER_SAMPLER_FNC
license: |
  Copyright (c) 2022 Patricio Gonzalez Vivo.
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.    
*/

#ifndef FLUIDSOLVER_DT
#define FLUIDSOLVER_DT 0.15
#endif

#ifndef FLUIDSOLVER_DX
#define FLUIDSOLVER_DX 1.0
#endif

// higher this threshold, lower the viscosity (max .8)
#ifndef FLUIDSOLVER_VISCOSITY
#define FLUIDSOLVER_VISCOSITY .16
#endif

// #ifndef FLUIDSOLVER_VELOCITY_DECAY
// #define FLUIDSOLVER_VELOCITY_DECAY 5e-6
// #endif 

#ifndef FLUIDSOLVER_SAMPLER_FNC
#define FLUIDSOLVER_SAMPLER_FNC(UV) texture2D(tex, UV)
#endif

#ifndef FNC_FLUIDSOLVER
#define FNC_FLUIDSOLVER

vec4 fluidSolver_sampler(sampler2D tex, vec2 st) {
    vec4 data = FLUIDSOLVER_SAMPLER_FNC(st);
    // XY velocity  -1.0 to 1.0
    // Z density     0.0 to 1.0
    // W vorticity   0.0 to 1.0
    data.xyw = data.xyw * 2.0 - 1.0;
    return data;
}

vec4 fluidSolver(sampler2D tex, vec2 st, vec2 pixel, vec2 force) {
    const float k = .2;
    const float s = k/FLUIDSOLVER_DT;
    const float dx = FLUIDSOLVER_DX;
    
    // Data
    vec4 d = fluidSolver_sampler(tex, st);
    vec4 dR = fluidSolver_sampler(tex, st + vec2(pixel.x, 0.));
    vec4 dL = fluidSolver_sampler(tex, st - vec2(pixel.x, 0.));
    vec4 dT = fluidSolver_sampler(tex, st + vec2(0., pixel.y));
    vec4 dB = fluidSolver_sampler(tex, st - vec2(0., pixel.y));

    // Delta Data
    vec4 ddx = (dR - dL).xyzw; // delta data on X
    vec4 ddy = (dT - dB).xyzw; // delta data on Y
    float divergence = (ddx.x + ddy.y) * 0.5;
    divergence = (ddx.x + ddy.y) / (2.0 * dx * dx);

    // Solving for density with one jacobi iteration 
    float a = 1.0 / (dx * dx);
    d.z = 1.0 / ( -4.0 * a ) * ( divergence - a * (dT.z + dR.z + dB.z + dL.z));
    
    #ifdef FLUIDSOLVER_VISCOSITY
    // Solving for velocity
    vec2 laplacian = dR.xy + dL.xy + dT.xy + dB.xy - 4.0 * d.xy;
    vec2 viscosityForce = FLUIDSOLVER_VISCOSITY * laplacian;
    // Semi-lagrangian advection
    vec2 densityInvariance = s * vec2(ddx.z, ddy.z);
    vec2 was = st - FLUIDSOLVER_DT * d.xy * pixel;
    d.xyw = fluidSolver_sampler(tex, was).xyw;
    d.xy += FLUIDSOLVER_DT * (viscosityForce - densityInvariance + force);
    #endif
    
    #ifdef FLUIDSOLVER_VELOCITY_DECAY
    d.xy = max(vec2(0.), abs(d.xy) - FLUIDSOLVER_VELOCITY_DECAY) * sign(d.xy);
    #endif
    
    // Vorticity confinement
    #ifdef FLUIDSOLVER_VORTICITY
    d.w = (dB.x - dT.x + dR.y - dL.y); // curl stored in the w channel
    vec2 vorticity = vec2(abs(dT.w) - abs(dB.w), abs(dL.w) - abs(dR.w));
    vorticity *= FLUIDSOLVER_VORTICITY/(length(vorticity) + 1e-5) * d.w;
    d.xy += vorticity;
    #endif

    // Boundary conditions
    d.xy *= smoothstep(0.5, 0.49,abs(st - 0.5));

    // Pack XY, Z and W data
    d.xy = clamp(d.xy, -0.999, 0.999) * 0.5 + 0.5;
    d.zw = saturate(d.zw);
    return d;
}

#endif