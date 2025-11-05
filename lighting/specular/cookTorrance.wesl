import lygia::lighting::common::ggx::GGXPrecise;
import lygia::lighting::common::smithGGXCorrelated::{smithGGXCorrelated, smithGGXCorrelated_Fast};
import lygia::math::saturate::saturate;
import lygia::lighting::fresnel::fresnel;

fn specularCookTorrance(L: vec3f, N: vec3f, H: vec3f, NoV: f32, NoL: f32, NoH: f32, linearRoughness: f32, specularColor: vec3f) -> vec3f {
    let LoH = saturate(dot(L, H));

    let D = GGXPrecise(N, H, NoH, linearRoughness);

    @if(TARGET_MOBILE) 
      let V = smithGGXCorrelated_Fast(NoV, NoL, linearRoughness);
    @else 
      let V = smithGGXCorrelated(NoV, NoL, linearRoughness);
    

    let F = fresnel(specularColor, LoH);

    return (D * V) * F;
}

// Note: ShadingData struct overload deferred until LATER (struct not yet defined in WESL)
