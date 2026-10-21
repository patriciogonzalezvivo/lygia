#include "../math/saturate.glsl"
#include "../color/luminance.glsl"
#include "../sampler.glsl"

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
    (its alpha) the density relaxes towards the source colour's density (dark = heavy by default),
    elsewhere it is left to the flow, so fluid leaving the source keeps its buoyancy.
    Use rayleighTaylorInit(source, st) to reset the state from that same source.
use:
    - <vec4> rayleighTaylor(<SAMPLER_TYPE> tex, <vec2> st, <vec2> pixel [, <vec2> force [, <float> density, <float> densityPct]])
    - <vec4> rayleighTaylor(<SAMPLER_TYPE> tex, <SAMPLER_TYPE> source, <vec2> st, <vec2> pixel, <vec2> force)
    - <vec4> rayleighTaylorInit(<SAMPLER_TYPE> source, <vec2> st)
options:
    - SAMPLER_FNC(TEX, UV): optional depending the target version of GLSL (texture2D(...) or texture(...))
    - RAYLEIGHTAYLOR_SAMPLER_FNC(TEX, UV): sampler function returning the decoded state. Default, SAMPLER_FNC(TEX, UV)
    - RAYLEIGHTAYLOR_GRAVITY: gravity vector in pixels per frame^2. Default, vec2(0.0, -0.06)
    - RAYLEIGHTAYLOR_ATWOOD: Atwood number (rho_heavy - rho_light) / (rho_heavy + rho_light). Default, 0.8
    - RAYLEIGHTAYLOR_MEAN_RADIUS: radius in pixels of the neighborhood buoyancy is measured against. Default, 12.0
    - RAYLEIGHTAYLOR_VISCOSITY: Default, 0.12
    - RAYLEIGHTAYLOR_DIFFUSION: density diffusion, keeps the interface a few pixels wide. Default, 0.02
    - RAYLEIGHTAYLOR_DAMPING: velocity damping per frame. Default, 0.995
    - RAYLEIGHTAYLOR_BOUNDARY: define it to apply no-slip walls on the edges of the texture
    - RAYLEIGHTAYLOR_SOURCE_SAMPLER_FNC(TEX, UV): sampler function for the source texture. Default, SAMPLER_FNC(TEX, UV)
    - RAYLEIGHTAYLOR_SOURCE_DENSITY_FNC(COLOR): density of a source colour. Default, (1.0 - luminance(COLOR.rgb))
    - RAYLEIGHTAYLOR_SOURCE_PCT: how much the density relaxes towards the source per frame where it fully covers. Default, 0.03
notes:
    - force is applied only where both fluids mix (density around 0.5), which is where it seeds the instability
    - density and densityPct relax the density towards a source value every frame, so the flow never fully mixes out
    - velocities are expected to stay under 1 pixel per frame when stored in a [0,1] encoded buffer
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#ifndef RAYLEIGHTAYLOR_SAMPLER_FNC
#define RAYLEIGHTAYLOR_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif

#ifndef RAYLEIGHTAYLOR_GRAVITY
#define RAYLEIGHTAYLOR_GRAVITY vec2(0.0, -0.06)
#endif

#ifndef RAYLEIGHTAYLOR_ATWOOD
#define RAYLEIGHTAYLOR_ATWOOD 0.8
#endif

#ifndef RAYLEIGHTAYLOR_MEAN_RADIUS
#define RAYLEIGHTAYLOR_MEAN_RADIUS 12.0
#endif

#ifndef RAYLEIGHTAYLOR_VISCOSITY
#define RAYLEIGHTAYLOR_VISCOSITY 0.12
#endif

#ifndef RAYLEIGHTAYLOR_DIFFUSION
#define RAYLEIGHTAYLOR_DIFFUSION 0.02
#endif

#ifndef RAYLEIGHTAYLOR_DAMPING
#define RAYLEIGHTAYLOR_DAMPING 0.995
#endif

#ifndef RAYLEIGHTAYLOR_SOURCE_SAMPLER_FNC
#define RAYLEIGHTAYLOR_SOURCE_SAMPLER_FNC(TEX, UV) SAMPLER_FNC(TEX, UV)
#endif

#ifndef RAYLEIGHTAYLOR_SOURCE_DENSITY_FNC
#define RAYLEIGHTAYLOR_SOURCE_DENSITY_FNC(COLOR) (1.0 - luminance(COLOR.rgb))
#endif

#ifndef RAYLEIGHTAYLOR_SOURCE_PCT
#define RAYLEIGHTAYLOR_SOURCE_PCT 0.03
#endif

#ifndef FNC_RAYLEIGHTAYLOR
#define FNC_RAYLEIGHTAYLOR

vec4 rayleighTaylor(SAMPLER_TYPE tex, vec2 st, vec2 pixel, vec2 force, float density, float densityPct) {
    vec4 d = RAYLEIGHTAYLOR_SAMPLER_FNC(tex, st);
    vec4 L = RAYLEIGHTAYLOR_SAMPLER_FNC(tex, st - vec2(pixel.x, 0.0));
    vec4 R = RAYLEIGHTAYLOR_SAMPLER_FNC(tex, st + vec2(pixel.x, 0.0));
    vec4 B = RAYLEIGHTAYLOR_SAMPLER_FNC(tex, st - vec2(0.0, pixel.y));
    vec4 T = RAYLEIGHTAYLOR_SAMPLER_FNC(tex, st + vec2(0.0, pixel.y));

    // Incompressibility: one Jacobi step on the pressure, warm-started
    // from the previous frame, then subtract its gradient
    float div = 0.5 * ((R.x - L.x) + (T.y - B.y));
    float p = (L.w + R.w + B.w + T.w - div) * 0.25;
    vec2 gradP = 0.5 * vec2(R.w - L.w, T.w - B.w);

    // Semi-Lagrangian advection of velocity and density
    vec4 prev = RAYLEIGHTAYLOR_SAMPLER_FNC(tex, st - d.xy * pixel);
    vec2 vel = prev.xy;
    float rho = prev.z;

    // Viscosity and density diffusion
    vel += RAYLEIGHTAYLOR_VISCOSITY * (L.xy + R.xy + B.xy + T.xy - 4.0 * d.xy);
    rho = mix(rho, (L.z + R.z + B.z + T.z) * 0.25, RAYLEIGHTAYLOR_DIFFUSION);

    // Buoyancy: heavier than the surroundings falls with gravity, lighter rises
    vec2 r = RAYLEIGHTAYLOR_MEAN_RADIUS * pixel;
    float rhoMean = 0.25 * (RAYLEIGHTAYLOR_SAMPLER_FNC(tex, st + vec2(r.x, 0.0)).z +
                            RAYLEIGHTAYLOR_SAMPLER_FNC(tex, st - vec2(r.x, 0.0)).z +
                            RAYLEIGHTAYLOR_SAMPLER_FNC(tex, st + vec2(0.0, r.y)).z +
                            RAYLEIGHTAYLOR_SAMPLER_FNC(tex, st - vec2(0.0, r.y)).z);
    vel += RAYLEIGHTAYLOR_GRAVITY * RAYLEIGHTAYLOR_ATWOOD * (rho - rhoMean) * 2.0;

    // External force, only where both fluids mix
    vel += force * 4.0 * rho * (1.0 - rho);

    vel -= gradP;

    // Relax the density towards the source
    rho = mix(rho, density, densityPct);

    #ifdef RAYLEIGHTAYLOR_BOUNDARY
    vec2 wall = 1.0 - smoothstep(vec2(0.5) - 2.0 * pixel, vec2(0.5), abs(st - 0.5));
    vel *= wall.x * wall.y;
    #endif

    return vec4(vel * RAYLEIGHTAYLOR_DAMPING, saturate(rho), p);
}

vec4 rayleighTaylor(SAMPLER_TYPE tex, vec2 st, vec2 pixel, vec2 force) {
    return rayleighTaylor(tex, st, pixel, force, 0.0, 0.0);
}

vec4 rayleighTaylor(SAMPLER_TYPE tex, vec2 st, vec2 pixel) {
    return rayleighTaylor(tex, st, pixel, vec2(0.0), 0.0, 0.0);
}

vec4 rayleighTaylor(SAMPLER_TYPE tex, SAMPLER_TYPE source, vec2 st, vec2 pixel, vec2 force) {
    vec4 src = RAYLEIGHTAYLOR_SOURCE_SAMPLER_FNC(source, st);
    return rayleighTaylor(tex, st, pixel, force, RAYLEIGHTAYLOR_SOURCE_DENSITY_FNC(src), saturate(src.a) * RAYLEIGHTAYLOR_SOURCE_PCT);
}

// At rest, with the source's density where it covers and neutral (0.5) elsewhere
vec4 rayleighTaylorInit(SAMPLER_TYPE source, vec2 st) {
    vec4 src = RAYLEIGHTAYLOR_SOURCE_SAMPLER_FNC(source, st);
    return vec4(0.0, 0.0, mix(0.5, RAYLEIGHTAYLOR_SOURCE_DENSITY_FNC(src), saturate(src.a)), 0.0);
}

#endif
