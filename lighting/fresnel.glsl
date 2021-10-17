#include "common/schlick.glsl"
#include "envMap.glsl"
#include "fakeCube.glsl"
#include "sphericalHarmonics.glsl"
#include "../color/tonemap.glsl"
#include "../math/saturate.glsl"

#ifndef FNC_FRESNEL
#define FNC_FRESNEL

vec3 fresnel(const vec3 f0, float LoH) {
#if defined(TARGET_MOBILE) || defined(PLATFORM_RPI) || defined(PLATFORM_WEBGL)
    return schlick(f0, 1.0, LoH);
#else
    float f90 = saturate(dot(f0, vec3(50.0 * 0.33)));
    return schlick(f0, f90, LoH);
#endif
}

vec3 fresnel(vec3 _R, vec3 _f0, float _NoV) {
    vec3 frsnl = fresnel(_f0, _NoV);
    // frsnl = schlick(_f0, vec3(1.), _NoV);

    vec3 reflectColor = vec3(0.0);
    #if defined(SCENE_SH_ARRAY)
    reflectColor = tonemap( sphericalHarmonics(_R) );
    #else
    reflectColor = fakeCube(_R);
    #endif

    return reflectColor * frsnl;
}

#endif