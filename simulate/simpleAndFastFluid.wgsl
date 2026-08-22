#include "../math/saturate.wgsl"
#include "../sampler.wgsl"

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

// higher this threshold, lower the viscosity (max .8)
// #define SIMPLEANDFASTFLUID_VISCOSITY .16

// #ifndef SIMPLEANDFASTFLUID_VELOCITY_DECAY
// #define SIMPLEANDFASTFLUID_VELOCITY_DECAY 5e-6
// #endif 

// #define SIMPLEANDFASTFLUID_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)

fn simpleAndFastFluid(tex: SAMPLER_TYPE, st: vec2f, pixel: vec2f) -> vec4f {
    const SIMPLEANDFASTFLUID_DT: f32 = 0.15;
    const SIMPLEANDFASTFLUID_DX: f32 = 1.0;
    let k = .2;
    let s = k/SIMPLEANDFASTFLUID_DT;
    let dx = SIMPLEANDFASTFLUID_DX;
    
    // Data
    let d = SIMPLEANDFASTFLUID_SAMPLER_FNC(tex, st);
    let dR = SIMPLEANDFASTFLUID_SAMPLER_FNC(tex, st + vec2f(pixel.x, 0.));
    let dL = SIMPLEANDFASTFLUID_SAMPLER_FNC(tex, st - vec2f(pixel.x, 0.));
    let dT = SIMPLEANDFASTFLUID_SAMPLER_FNC(tex, st + vec2f(0., pixel.y));
    let dB = SIMPLEANDFASTFLUID_SAMPLER_FNC(tex, st - vec2f(0., pixel.y));

    // Delta Data
    vec4 ddx = (dR - dL).xyzw; // delta data on X
    vec4 ddy = (dT - dB).xyzw; // delta data on Y
    let divergence = (ddx.x + ddy.y) * 0.5;
    divergence = (ddx.x + ddy.y) / (2.0 * dx * dx);

    // Solving for density with one jacobi iteration 
    let a = 1.0 / (dx * dx);
    d.z = 1.0 / ( -4.0 * a ) * ( divergence - a * (dT.z + dR.z + dB.z + dL.z));
    
    // Solving for velocity
    let laplacian = dR.xy + dL.xy + dT.xy + dB.xy - 4.0 * d.xy;
    let viscosityForce = SIMPLEANDFASTFLUID_VISCOSITY * laplacian;
    // Semi-lagrangian advection
    let densityInvariance = s * vec2f(ddx.z, ddy.z);
    let was = st - SIMPLEANDFASTFLUID_DT * d.xy * pixel;
    d.xyw = SIMPLEANDFASTFLUID_SAMPLER_FNC(tex, was).xyw;
    d.xy += SIMPLEANDFASTFLUID_DT * (viscosityForce - densityInvariance);
    
    d.xy = max(vec2f(0.), abs(d.xy) - SIMPLEANDFASTFLUID_VELOCITY_DECAY) * sign(d.xy);
    
    // Vorticity confinement
    d.w = (dB.x - dT.x + dR.y - dL.y); // curl stored in the w channel
    let vorticity = vec2f(abs(dT.w) - abs(dB.w), abs(dL.w) - abs(dR.w));
    vorticity *= SIMPLEANDFASTFLUID_VORTICITY/(length(vorticity) + 1e-5) * d.w;
    d.xy += vorticity;

    // Boundary conditions
    d.xy *= smoothstep(0.5, 0.49,abs(st - 0.5));

    // Pack XY, Z and W data
    d.xy = clamp(d.xy, -0.999, 0.999);
    d.zw = saturate(d.zw);
    return d;
}

fn simpleAndFastFluida(tex: SAMPLER_TYPE, st: vec2f, pixel: vec2f, force: vec2f) -> vec4f {
    const SIMPLEANDFASTFLUID_DT: f32 = 0.15;
    const SIMPLEANDFASTFLUID_DX: f32 = 1.0;
    let k = .2;
    let s = k/SIMPLEANDFASTFLUID_DT;
    let dx = SIMPLEANDFASTFLUID_DX;
    
    // Data
    let d = SIMPLEANDFASTFLUID_SAMPLER_FNC(tex, st);
    let dR = SIMPLEANDFASTFLUID_SAMPLER_FNC(tex, st + vec2f(pixel.x, 0.));
    let dL = SIMPLEANDFASTFLUID_SAMPLER_FNC(tex, st - vec2f(pixel.x, 0.));
    let dT = SIMPLEANDFASTFLUID_SAMPLER_FNC(tex, st + vec2f(0., pixel.y));
    let dB = SIMPLEANDFASTFLUID_SAMPLER_FNC(tex, st - vec2f(0., pixel.y));

    // Delta Data
    vec4 ddx = (dR - dL).xyzw; // delta data on X
    vec4 ddy = (dT - dB).xyzw; // delta data on Y
    let divergence = (ddx.x + ddy.y) * 0.5;
    divergence = (ddx.x + ddy.y) / (2.0 * dx * dx);

    // Solving for density with one jacobi iteration 
    let a = 1.0 / (dx * dx);
    d.z = 1.0 / ( -4.0 * a ) * ( divergence - a * (dT.z + dR.z + dB.z + dL.z));
    
    // Solving for velocity
    let laplacian = dR.xy + dL.xy + dT.xy + dB.xy - 4.0 * d.xy;
    let viscosityForce = SIMPLEANDFASTFLUID_VISCOSITY * laplacian;
    // Semi-lagrangian advection
    let densityInvariance = s * vec2f(ddx.z, ddy.z);
    let was = st - SIMPLEANDFASTFLUID_DT * d.xy * pixel;
    d.xyw = SIMPLEANDFASTFLUID_SAMPLER_FNC(tex, was).xyw;
    d.xy += SIMPLEANDFASTFLUID_DT * (viscosityForce - densityInvariance + force);
    
    d.xy = max(vec2f(0.), abs(d.xy) - SIMPLEANDFASTFLUID_VELOCITY_DECAY) * sign(d.xy);
    
    // Vorticity confinement
    d.w = (dB.x - dT.x + dR.y - dL.y); // curl stored in the w channel
    let vorticity = vec2f(abs(dT.w) - abs(dB.w), abs(dL.w) - abs(dR.w));
    vorticity *= SIMPLEANDFASTFLUID_VORTICITY/(length(vorticity) + 1e-5) * d.w;
    d.xy += vorticity;

    // Boundary conditions
    d.xy *= smoothstep(0.5, 0.49,abs(st - 0.5));

    // Pack XY, Z and W data
    d.xy = clamp(d.xy, -0.999, 0.999);
    d.zw = saturate(d.zw);
    return d;
}
