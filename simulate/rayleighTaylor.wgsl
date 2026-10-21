#include "../math/saturate.wgsl"
#include "../color/luminance.wgsl"
#include "../sampler.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: |
    Single pass Rayleigh-Taylor instability: a heavy fluid over a light one under gravity, using the
    Boussinesq approximation. Velocity and density are semi-Lagrangian advected, incompressibility is
    approximated with one warm-started Jacobi pressure iteration per frame and buoyancy is measured
    against the local mean density, so only horizontal density gradients (baroclinic torque) spin
    up the characteristic sinking fingers and rising mushroom plumes. https://en.wikipedia.org/wiki/Rayleigh%E2%80%93Taylor_instability
    The reading texture is structured as follows:
        XY velocity in pixels per frame
        Z  density of the heavy fluid (0.0 light .. 1.0 heavy)
        W  pressure
    The result is returned in the same layout (decoded), so encode it before writing it back if your buffer is not a float one.
    Density can also be fed from a source texture (e.g. a rendered scene): where the source covers
    (its alpha) the density relaxes towards the source colour's density (dark = heavy),
    elsewhere it is left to the flow, so fluid leaving the source keeps its buoyancy.
use:
    - <vec4> rayleighTaylor(<SAMPLER_TYPE> tex, <vec2> st, <vec2> pixel, <vec2> force, <float> density, <float> densityPct)
    - <vec4> rayleighTaylorSource(<SAMPLER_TYPE> tex, <SAMPLER_TYPE> source, <vec2> st, <vec2> pixel, <vec2> force)
    - <vec4> rayleighTaylorInit(<SAMPLER_TYPE> source, <vec2> st)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - RAYLEIGHTAYLOR_SAMPLER_FNC(TEX, UV): sampler function returning the decoded state
    - RAYLEIGHTAYLOR_SOURCE_SAMPLER_FNC(TEX, UV): sampler function for the source texture
notes:
    - WGSL has no function overloading, so force, density and densityPct are always required (pass zeros to disable them)
      and the source-driven variant is named rayleighTaylorSource
    - The RAYLEIGHTAYLOR_* tuning options of the GLSL version are constants here
    - RAYLEIGHTAYLOR_BOUNDARY is always applied
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

// #define RAYLEIGHTAYLOR_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
// #define RAYLEIGHTAYLOR_SOURCE_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)

const RAYLEIGHTAYLOR_GRAVITY: vec2f = vec2f(0.0, -0.06);
const RAYLEIGHTAYLOR_ATWOOD: f32 = 0.8;
const RAYLEIGHTAYLOR_MEAN_RADIUS: f32 = 12.0;
const RAYLEIGHTAYLOR_VISCOSITY: f32 = 0.12;
const RAYLEIGHTAYLOR_DIFFUSION: f32 = 0.02;
const RAYLEIGHTAYLOR_DAMPING: f32 = 0.995;
const RAYLEIGHTAYLOR_SOURCE_PCT: f32 = 0.03;

fn rayleighTaylor(tex: SAMPLER_TYPE, st: vec2f, pixel: vec2f, force: vec2f, density: f32, densityPct: f32) -> vec4f {
    let d = RAYLEIGHTAYLOR_SAMPLER_FNC(tex, st);
    let L = RAYLEIGHTAYLOR_SAMPLER_FNC(tex, st - vec2f(pixel.x, 0.0));
    let R = RAYLEIGHTAYLOR_SAMPLER_FNC(tex, st + vec2f(pixel.x, 0.0));
    let B = RAYLEIGHTAYLOR_SAMPLER_FNC(tex, st - vec2f(0.0, pixel.y));
    let T = RAYLEIGHTAYLOR_SAMPLER_FNC(tex, st + vec2f(0.0, pixel.y));

    // Incompressibility: one Jacobi step on the pressure, warm-started
    // from the previous frame, then subtract its gradient
    let div = 0.5 * ((R.x - L.x) + (T.y - B.y));
    let p = (L.w + R.w + B.w + T.w - div) * 0.25;
    let gradP = 0.5 * vec2f(R.w - L.w, T.w - B.w);

    // Semi-Lagrangian advection of velocity and density
    let prev = RAYLEIGHTAYLOR_SAMPLER_FNC(tex, st - d.xy * pixel);
    var vel = prev.xy;
    var rho = prev.z;

    // Viscosity and density diffusion
    vel += RAYLEIGHTAYLOR_VISCOSITY * (L.xy + R.xy + B.xy + T.xy - 4.0 * d.xy);
    rho = mix(rho, (L.z + R.z + B.z + T.z) * 0.25, RAYLEIGHTAYLOR_DIFFUSION);

    // Buoyancy: heavier than the surroundings falls with gravity, lighter rises
    let r = RAYLEIGHTAYLOR_MEAN_RADIUS * pixel;
    let rhoMean = 0.25 * (RAYLEIGHTAYLOR_SAMPLER_FNC(tex, st + vec2f(r.x, 0.0)).z +
                          RAYLEIGHTAYLOR_SAMPLER_FNC(tex, st - vec2f(r.x, 0.0)).z +
                          RAYLEIGHTAYLOR_SAMPLER_FNC(tex, st + vec2f(0.0, r.y)).z +
                          RAYLEIGHTAYLOR_SAMPLER_FNC(tex, st - vec2f(0.0, r.y)).z);
    vel += RAYLEIGHTAYLOR_GRAVITY * RAYLEIGHTAYLOR_ATWOOD * (rho - rhoMean) * 2.0;

    // External force, only where both fluids mix
    vel += force * 4.0 * rho * (1.0 - rho);

    vel -= gradP;

    // Relax the density towards the source
    rho = mix(rho, density, densityPct);

    // Boundary conditions
    let wall = 1.0 - smoothstep(vec2f(0.5) - 2.0 * pixel, vec2f(0.5), abs(st - 0.5));
    vel *= wall.x * wall.y;

    return vec4f(vel * RAYLEIGHTAYLOR_DAMPING, saturate(rho), p);
}

fn rayleighTaylorSource(tex: SAMPLER_TYPE, source: SAMPLER_TYPE, st: vec2f, pixel: vec2f, force: vec2f) -> vec4f {
    let src = RAYLEIGHTAYLOR_SOURCE_SAMPLER_FNC(source, st);
    return rayleighTaylor(tex, st, pixel, force, 1.0 - luminance3(src.rgb), saturate(src.a) * RAYLEIGHTAYLOR_SOURCE_PCT);
}

// At rest, with the source's density where it covers and neutral (0.5) elsewhere
fn rayleighTaylorInit(source: SAMPLER_TYPE, st: vec2f) -> vec4f {
    let src = RAYLEIGHTAYLOR_SOURCE_SAMPLER_FNC(source, st);
    return vec4f(0.0, 0.0, mix(0.5, 1.0 - luminance3(src.rgb), saturate(src.a)), 0.0);
}
