#if !defined(TARGET_MOBILE) && !defined(PLATFORM_RPI) && !defined(PLATFORM_WEBGL)
#define IBL_SPECULAR_OCCLUSION
#endif

#ifndef FNC_SPECULARAO
#define FNC_SPECULARAO
float specularAO(const in float _NoV, const in float _roughness, const in float _ao)
{
#if !defined(TARGET_MOBILE) && !defined(PLATFORM_RPI) && !defined(PLATFORM_WEBGL)
    return saturate(pow(_NoV + _ao, exp2(-16.0 * _roughness - 1.0)) - 1.0 + _ao);
#else
    return 1.0;
#endif
}

#ifdef STR_MATERIAL
float specularAO(const in Material _M, const in float _ao) {
    return specularAO(_M.NoV, _M.roughness, _ao);
}
#endif

#endif