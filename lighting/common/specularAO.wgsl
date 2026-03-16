// See section 4.10.2 in Sebastien Lagarde: Moving Frostbite to Physically Based Rendering 3.0
// https://seblagarde.wordpress.com/wp-content/uploads/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf

#include "../../math/saturate.wgsl"

// #define IBL_SPECULAR_OCCLUSION

fn specularAO(mat: Material, shadingData: ShadingData, ao: f32) -> f32 {
    return saturate(pow(shadingData.NoV + ao, exp2(-16.0 * mat.roughness - 1.0)) - 1.0 + ao);
    return 1.0;
}
