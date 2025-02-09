#include "../math/saturate.glsl"
#include "../sampler.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: |
    Simple single pass fluid simulation from the book GPU Pro 2, "Simple and Fast Fluids" . https://inria.hal.science/inria-00596050/document
    The algorithm uses a Jacobi iteration method to solve for the density and incorporates semi-Lagrangian advection

use: <vec2> simpleAndFastFluid(<SAMPLER_TYPE> tex, <vec2> st, <vec2> pixel, <vec2> force)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - SIMPLEANDFASTFLUID_DT: Default, 0.15
    - SIMPLEANDFASTFLUID_DX: Default, 1.0
    - SIMPLEANDFASTFLUID_BOUNDARY: apply set to true if you want to use the boundary conditions
    - SIMPLEANDFASTFLUID_VORTICITY: lower value for vorticity threshold means higher viscosity and vice versa (max .3)
    - SIMPLEANDFASTFLUID_VISCOSITY: Default, 0.16
    - SIMPLEANDFASTFLUID_VELOCITY_DECAY: Default, 5e-6
    - SIMPLEANDFASTFLUID_SAMPLER_FNC: sampler function
*/

#ifndef SIMPLEANDFASTFLUID_DT
#define SIMPLEANDFASTFLUID_DT 0.15
#endif

#ifndef SIMPLEANDFASTFLUID_DX
#define SIMPLEANDFASTFLUID_DX 1.0
#endif

// higher this threshold, lower the viscosity (max .8)
#ifndef SIMPLEANDFASTFLUID_VISCOSITY
#define SIMPLEANDFASTFLUID_VISCOSITY .16
#endif

// #ifndef SIMPLEANDFASTFLUID_VELOCITY_DECAY
// #define SIMPLEANDFASTFLUID_VELOCITY_DECAY 5e-6
// #endif 

#ifndef SIMPLEANDFASTFLUID_SAMPLER_FNC
#define SIMPLEANDFASTFLUID_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif

#ifndef FNC_SIMPLEANDFASTFLUID
#define FNC_SIMPLEANDFASTFLUID

vec4 simpleAndFastFluid(SAMPLER_TYPE tex, vec2 st, vec2 pixel) {
    const float k = .2;
    const float s = k/SIMPLEANDFASTFLUID_DT;
    const float dx = SIMPLEANDFASTFLUID_DX;
    
    // Data
    vec4 d = SIMPLEANDFASTFLUID_SAMPLER_FNC(tex, st);
    vec4 dR = SIMPLEANDFASTFLUID_SAMPLER_FNC(tex, st + vec2(pixel.x, 0.));
    vec4 dL = SIMPLEANDFASTFLUID_SAMPLER_FNC(tex, st - vec2(pixel.x, 0.));
    vec4 dT = SIMPLEANDFASTFLUID_SAMPLER_FNC(tex, st + vec2(0., pixel.y));
    vec4 dB = SIMPLEANDFASTFLUID_SAMPLER_FNC(tex, st - vec2(0., pixel.y));

    // Delta Data
    vec4 ddx = (dR - dL).xyzw; // delta data on X
    vec4 ddy = (dT - dB).xyzw; // delta data on Y
    float divergence = (ddx.x + ddy.y) * 0.5;
    divergence = (ddx.x + ddy.y) / (2.0 * dx * dx);

    // Solving for density with one jacobi iteration 
    float a = 1.0 / (dx * dx);
    d.z = 1.0 / ( -4.0 * a ) * ( divergence - a * (dT.z + dR.z + dB.z + dL.z));
    
    #ifdef SIMPLEANDFASTFLUID_VISCOSITY
    // Solving for velocity
    vec2 laplacian = dR.xy + dL.xy + dT.xy + dB.xy - 4.0 * d.xy;
    vec2 viscosityForce = SIMPLEANDFASTFLUID_VISCOSITY * laplacian;
    // Semi-lagrangian advection
    vec2 densityInvariance = s * vec2(ddx.z, ddy.z);
    vec2 was = st - SIMPLEANDFASTFLUID_DT * d.xy * pixel;
    d.xyw = SIMPLEANDFASTFLUID_SAMPLER_FNC(tex, was).xyw;
    d.xy += SIMPLEANDFASTFLUID_DT * (viscosityForce - densityInvariance);
    #endif
    
    #ifdef SIMPLEANDFASTFLUID_VELOCITY_DECAY
    d.xy = max(vec2(0.), abs(d.xy) - SIMPLEANDFASTFLUID_VELOCITY_DECAY) * sign(d.xy);
    #endif
    
    // Vorticity confinement
    #ifdef SIMPLEANDFASTFLUID_VORTICITY
    d.w = (dB.x - dT.x + dR.y - dL.y); // curl stored in the w channel
    vec2 vorticity = vec2(abs(dT.w) - abs(dB.w), abs(dL.w) - abs(dR.w));
    vorticity *= SIMPLEANDFASTFLUID_VORTICITY/(length(vorticity) + 1e-5) * d.w;
    d.xy += vorticity;
    #endif

    #ifdef SIMPLEANDFASTFLUID_BOUNDARY
    // Boundary conditions
    d.xy *= smoothstep(0.5, 0.49,abs(st - 0.5));
    #endif

    // Pack XY, Z and W data
    d.xy = clamp(d.xy, -0.999, 0.999);
    d.zw = saturate(d.zw);
    return d;
}

vec4 simpleAndFastFluid(SAMPLER_TYPE tex, vec2 st, vec2 pixel, vec2 force) {
    const float k = .2;
    const float s = k/SIMPLEANDFASTFLUID_DT;
    const float dx = SIMPLEANDFASTFLUID_DX;
    
    // Data
    vec4 d = SIMPLEANDFASTFLUID_SAMPLER_FNC(tex, st);
    vec4 dR = SIMPLEANDFASTFLUID_SAMPLER_FNC(tex, st + vec2(pixel.x, 0.));
    vec4 dL = SIMPLEANDFASTFLUID_SAMPLER_FNC(tex, st - vec2(pixel.x, 0.));
    vec4 dT = SIMPLEANDFASTFLUID_SAMPLER_FNC(tex, st + vec2(0., pixel.y));
    vec4 dB = SIMPLEANDFASTFLUID_SAMPLER_FNC(tex, st - vec2(0., pixel.y));

    // Delta Data
    vec4 ddx = (dR - dL).xyzw; // delta data on X
    vec4 ddy = (dT - dB).xyzw; // delta data on Y
    float divergence = (ddx.x + ddy.y) * 0.5;
    divergence = (ddx.x + ddy.y) / (2.0 * dx * dx);

    // Solving for density with one jacobi iteration 
    float a = 1.0 / (dx * dx);
    d.z = 1.0 / ( -4.0 * a ) * ( divergence - a * (dT.z + dR.z + dB.z + dL.z));
    
    #ifdef SIMPLEANDFASTFLUID_VISCOSITY
    // Solving for velocity
    vec2 laplacian = dR.xy + dL.xy + dT.xy + dB.xy - 4.0 * d.xy;
    vec2 viscosityForce = SIMPLEANDFASTFLUID_VISCOSITY * laplacian;
    // Semi-lagrangian advection
    vec2 densityInvariance = s * vec2(ddx.z, ddy.z);
    vec2 was = st - SIMPLEANDFASTFLUID_DT * d.xy * pixel;
    d.xyw = SIMPLEANDFASTFLUID_SAMPLER_FNC(tex, was).xyw;
    d.xy += SIMPLEANDFASTFLUID_DT * (viscosityForce - densityInvariance + force);
    #endif
    
    #ifdef SIMPLEANDFASTFLUID_VELOCITY_DECAY
    d.xy = max(vec2(0.), abs(d.xy) - SIMPLEANDFASTFLUID_VELOCITY_DECAY) * sign(d.xy);
    #endif
    
    // Vorticity confinement
    #ifdef SIMPLEANDFASTFLUID_VORTICITY
    d.w = (dB.x - dT.x + dR.y - dL.y); // curl stored in the w channel
    vec2 vorticity = vec2(abs(dT.w) - abs(dB.w), abs(dL.w) - abs(dR.w));
    vorticity *= SIMPLEANDFASTFLUID_VORTICITY/(length(vorticity) + 1e-5) * d.w;
    d.xy += vorticity;
    #endif

    #ifdef SIMPLEANDFASTFLUID_BOUNDARY
    // Boundary conditions
    d.xy *= smoothstep(0.5, 0.49,abs(st - 0.5));
    #endif

    // Pack XY, Z and W data
    d.xy = clamp(d.xy, -0.999, 0.999);
    d.zw = saturate(d.zw);
    return d;
}

#endif