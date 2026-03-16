#include "../common/beckmann.wgsl"

fn specularBeckmann(shadingData: ShadingData) -> f32 {
    return beckmann(shadingData.NoH, shadingData.roughness);
}
