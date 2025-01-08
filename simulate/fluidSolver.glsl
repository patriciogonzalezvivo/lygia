#include "../math/saturate.glsl"
#include "../sampler.glsl"

/*
original_author: Patricio Gonzalez Vivo
description: |
    Simple single pass fluid simlation from https://wyattflanders.com/MeAndMyNeighborhood.pdf
use: <vec2> fluidSolver(<SAMPLER_TYPE> tex, <vec2> st, <vec2> pixel, <vec2> force)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - FLUIDSOLVER_SAMPLER_FNC: sampler function
*/


#ifndef FLUIDSOLVER_SAMPLER_FNC
#define FLUIDSOLVER_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif

#ifndef FNC_FLUIDSOLVER
#define FNC_FLUIDSOLVER

vec4 fluidSolverDataSampler(sampler2D tex, vec2 st, vec2 pixel) { 
    vec2 offset = FLUIDSOLVER_SAMPLER_FNC(tex, st).xy;
    return FLUIDSOLVER_SAMPLER_FNC(tex, st - offset * pixel); 
}

vec4 fluidSolver(sampler2D tex, vec2 st, vec2 pixel) {
    vec4 d = fluidSolverDataSampler(tex, st, pixel);

    // Rule 1: define the neighborhood
    vec4 pX   = fluidSolverDataSampler(tex, st + vec2(pixel.x,0.0), pixel);
    vec4 pY   = fluidSolverDataSampler(tex, st + vec2(0.0,pixel.y), pixel);
    vec4 nX   = fluidSolverDataSampler(tex, st - vec2(pixel.x,0.0), pixel);
    vec4 nY   = fluidSolverDataSampler(tex, st - vec2(0.0,pixel.y), pixel);
    
    // Rule 2: Disordered color diffuses completely :
    d.b = (pX.b + pY.b + nX.b + nY.b) * 0.25;
    
    // Rule 3: Order in the disordered data creates Order :
    d.xy += vec2(nX.b - pX.b, nY.b - pY.b) * 0.25;
    
    // Rule 4: Disorder in the ordered data creates Disorder :
    d.b += (nX.x - pX.x + nY.y - pY.y) * 0.25;
    
    // Mass concervation:
    d.w += (nX.x * nX.w - pX.x * pX.w + nY.y * nY.w - pY.y * pY.w) * 0.25;

    #ifdef FLUIDSOLVER_BOUNDARY
    //Boundary conditions
    if (st.x < pixel.x || st.y < pixel.y || st.x > (1.0 - pixel.x) || st.y > (1.0 - pixel.y) ) { 
        d.xy *= 0.;
    }
    #endif

    return d;
}

vec4 fluidSolver(sampler2D tex, vec2 st, vec2 pixel, vec2 force) {
    vec4 d = fluidSolver(tex, st, pixel);
    d.xy += force * saturate(d.w) * pixel;
    // d.xy = clamp(d.xy, -0.9999, 0.9999);
    // d.zw = clamp(d.zw, 0.0001, 0.9999);
    return d;
}

#endif