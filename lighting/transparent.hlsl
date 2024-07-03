#include "envMap.hlsl"
#include "ior.hlsl"
#include "ior/2eta.hlsl"
#include "ior/2f0.hlsl"

/*
contributors: Patricio Gonzalez Vivo
description: This function simulates the refraction of light through a transparent material. It uses the Schlick's approximation to calculate the Fresnel reflection and the Snell's law to calculate the refraction. It also uses the envMap function to simulate the dispersion of light through the material.
use:
    - <float3> transparent(<float3> normal, <float3> view, <float3> ior, <float> roughness)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#if !defined(TRANSPARENT_DISPERSION) && defined(TRANSPARENT_DISPERSION_PASSES)
#define TRANSPARENT_DISPERSION 0.05
#elif defined(TRANSPARENT_DISPERSION) && !defined(TRANSPARENT_DISPERSION_PASSES)
#define TRANSPARENT_DISPERSION_PASSES 6
#endif

#ifndef FNC_TRANSPARENT
#define FNC_TRANSPARENT

float3 transparent(float3 normal, float3 view, float3 Fr, float3 eta, float roughness) {
    float3 color = float3(0.0, 0.0, 0.0);
    float3 T = max(float3(0.0, 0.0, 0.0), 1.0 - Fr);

    #if defined(TRANSPARENT_DISPERSION) && defined(TRANSPARENT_DISPERSION_PASSES)
        float pass_step = 1.0/float(TRANSPARENT_DISPERSION_PASSES);
        float3 bck = float3(0.0, 0.0, 0.0);
        for ( int i = 0; i < TRANSPARENT_DISPERSION_PASSES; i++ ) {
            float slide = float(i) * pass_step * TRANSPARENT_DISPERSION;
            float3 R      = refract(view, normal, eta.g );
            float3 ref    = envMap(R, roughness, 0.0);

            #if !defined(TRANSPARENT_DISPERSION_FAST) && !defined(TARGET_MOBILE) && !defined(PLATFORM_RPI)
            ref.r       = envMap(refract(view, normal, eta.r - slide), roughness, 0.0).r;
            ref.b       = envMap(refract(view, normal, eta.b + slide), roughness, 0.0).b;
            #endif

            bck += ref;
        }
        color.rgb   = bck * pass_step;
    #else 

        float3 R      = refract(view, normal, eta.g);
        color       = envMap(R, roughness);

        #if !defined(TRANSPARENT_DISPERSION_FAST) && !defined(TARGET_MOBILE) && !defined(PLATFORM_RPI)
        float3 RaR    = refract(view, normal, eta.r);
        float3 RaB    = refract(view, normal, eta.b);
        color.r     = envMap(RaR, roughness).r;
        color.b     = envMap(RaB, roughness).b;
        #endif

    #endif

    return color*T*T*T*T;
}

float3 transparent(float3 normal, float3 view, float Fr, float3 eta, float roughness) {
    float3 color = float3(0.0, 0.0, 0.0);
    float T = max(0.0, 1.0-Fr);

    #if defined(TRANSPARENT_DISPERSION) && defined(TRANSPARENT_DISPERSION_PASSES)
        float pass_step = 1.0/float(TRANSPARENT_DISPERSION_PASSES);
        float3 bck = float3(0.0, 0.0, 0.0);
        for ( int i = 0; i < TRANSPARENT_DISPERSION_PASSES; i++ ) {
            float slide = float(i) * pass_step * TRANSPARENT_DISPERSION;
            float3 R      = refract(view, normal, eta.g );
            float3 ref    = envMap(R, roughness, 0.0);

            #if !defined(TRANSPARENT_DISPERSION_FAST) && !defined(TARGET_MOBILE) && !defined(PLATFORM_RPI)
            ref.r       = envMap(refract(view, normal, eta.r - slide), roughness, 0.0).r;
            ref.b       = envMap(refract(view, normal, eta.b + slide), roughness, 0.0).b;
            #endif

            bck += ref;
        }
        color.rgb   = bck * pass_step;
    #else 

        float3 R      = refract(view, normal, eta.g);
        color       = envMap(R, roughness);

        #if !defined(TRANSPARENT_DISPERSION_FAST) && !defined(TARGET_MOBILE) && !defined(PLATFORM_RPI)
        float3 RaR    = refract(view, normal, eta.r);
        float3 RaB    = refract(view, normal, eta.b);
        color.r     = envMap(RaR, roughness).r;
        color.b     = envMap(RaB, roughness).b;
        #endif

    #endif

    return color*T*T*T*T;
}

#endif
