#include "../math/saturate.wgsl"
#include "../sampler.wgsl"

/*
original_author: Patricio Gonzalez Vivo
description: |
    Stable fluid simulation inspired on Lattice Boltzmann method described in https://wyattflanders.com/MeAndMyNeighborhood.pdf
    Where the reading texture is structure as follow
        XY is the ordered energy that (moves the bubble)
        B disordered energy that evaporates
        W internal mass (follow the bubble)

use: <vec2> LATTICEBOLTZMANN(<SAMPLER_TYPE> tex, <vec2> st, <vec2> pixel, <vec2> force)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - LATTICEBOLTZMANN_SAMPLER_FNC: sampler function
*/

// #define LATTICEBOLTZMANN_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)

// Rule 1: All My energy moves with me
//  I should find My Energy not where it is, but where it was.
fn latticeBoltzmannPrevPosSampler(tex: sampler2D, st: vec2f, pixel: vec2f) -> vec4f {
    let offset = LATTICEBOLTZMANN_SAMPLER_FNC(tex, st).xy;
    return LATTICEBOLTZMANN_SAMPLER_FNC(tex, st - offset * pixel); 
}

fn latticeBoltzmann(tex: sampler2D, st: vec2f, pixel: vec2f) -> vec4f {
    let d = latticeBoltzmannPrevPosSampler(tex, st, pixel);

    // Neighbors
    let pX = latticeBoltzmannPrevPosSampler(tex, st + vec2f(pixel.x,0.0), pixel);
    let pY = latticeBoltzmannPrevPosSampler(tex, st + vec2f(0.0,pixel.y), pixel);
    let nX = latticeBoltzmannPrevPosSampler(tex, st - vec2f(pixel.x,0.0), pixel);
    let nY = latticeBoltzmannPrevPosSampler(tex, st - vec2f(0.0,pixel.y), pixel);
    
    // Rule 2: Disordered diffuses completely.
    //  All of my disordered Energy B comes from my neighborhood.
    //  Exchange disorder symmetrically in all direction
    d.b = (pX.b + pY.b + nX.b + nY.b) * 0.25;
    
    // Rule 3: Order in the disordered data creates Order (XY)
    //  The change in volatile Energy B across me will push Me in that direction.
    d.xy += vec2f(nX.b - pX.b, nY.b - pY.b) * 0.25;
    
    // Rule 4: Disorder in the ordered data creates Disorder (B)
    //  The disorder in the order around Me, enters Me as disorder
    d.b += (nX.x - pX.x + nY.y - pY.y) * 0.25;
    
    // Mass concervation:
    d.w += (nX.x * nX.w - pX.x * pX.w + nY.y * nY.w - pY.y * pY.w) * 0.25;

    //Boundary conditions
    if (st.x < pixel.x || st.y < pixel.y || st.x > (1.0 - pixel.x) || st.y > (1.0 - pixel.y) ) { 
        d.xy *= 0.;
    }

    return d;
}

fn latticeBoltzmanna(tex: sampler2D, st: vec2f, pixel: vec2f, force: vec2f) -> vec4f {
    let d = latticeBoltzmann(tex, st, pixel);
    d.xy += force * saturate(d.w) * pixel;
    // d.xy = clamp(d.xy, -0.9999, 0.9999);
    // d.zw = clamp(d.zw, 0.0001, 0.9999);
    return d;
}
