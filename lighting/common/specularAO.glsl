// See section 4.10.2 in Sebastien Lagarde: Moving Frostbite to Physically Based Rendering 3.0
// https://seblagarde.wordpress.com/wp-content/uploads/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf

#include "../../math/saturate.glsl"

#if !defined(TARGET_MOBILE) && !defined(PLATFORM_RPI) && !defined(PLATFORM_WEBGL)
#define IBL_SPECULAR_OCCLUSION
#endif

#ifndef FNC_SPECULARAO
#define FNC_SPECULARAO
float specularAO(Material mat, ShadingData shadingData, const in float ao) {
#if !defined(TARGET_MOBILE) && !defined(PLATFORM_RPI) && !defined(PLATFORM_WEBGL)
    return saturate(pow(shadingData.NoV + ao, exp2(-16.0 * mat.roughness - 1.0)) - 1.0 + ao);
#else
    return 1.0;
#endif
}

#endif