#if !defined(TARGET_MOBILE) && !defined(PLATFORM_RPI) && !defined(PLATFORM_WEBGL)
#define IBL_SPECULAR_OCCLUSION
#endif

#ifndef FNC_SPECULARAO
#define FNC_SPECULARAO
float specularAO(float NoV, float ao, float roughness) {
#if !defined(TARGET_MOBILE) && !defined(PLATFORM_RPI) && !defined(PLATFORM_WEBGL)
    return saturate(pow(NoV + ao, exp2(-16.0 * roughness - 1.0)) - 1.0 + ao);
#else
    return 1.0;
#endif
}
#endif