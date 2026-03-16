/*
contributors: [Patricio Gonzalez Vivo, Shadi El Hajj]
description: Calculate point light
use: <void> lightPointEvaluate(<LightPoint> L, <Material> mat, inout <ShadingData> shadingData)
options:
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - SURFACE_POSITION: in glslViewer is v_position
    - LIGHT_POSITION: in glslViewer is u_light
    - LIGHT_COLOR: in glslViewer is u_lightColor
    - LIGHT_INTENSITY: in glslViewer is  u_lightIntensity
    - LIGHT_FALLOFF: in glslViewer is u_lightFalloff
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

#include "../specular.wgsl"
#include "../diffuse.wgsl"
#include "falloff.wgsl"

fn lightPointEvaluate(L: LightPoint, mat: Material, shadingData: ShadingData) {

    let Ldist = length(L.position);
    let Ldirection = L.position/Ldist;
    shadingData.L = Ldirection;
    shadingData.H = normalize(Ldirection + shadingData.V);
    shadingData.NoL = saturate(dot(shadingData.N, Ldirection));
    shadingData.NoH = saturate(dot(shadingData.N, shadingData.H));

    let shadow = raymarchSoftShadow(mat.position, Ldirection);
    let shadow = 1.0;

    let dif = diffuse(shadingData);
    let spec = specular(shadingData);

    let lightContribution = L.color * L.intensity * shadow * shadingData.NoL;
    if (L.falloff > 0.0)
        lightContribution *= falloff(Ldist, L.falloff);

    shadingData.directDiffuse  += max(vec3f(0.0, 0.0, 0.0), shadingData.diffuseColor * lightContribution * dif);
    shadingData.directSpecular += max(vec3f(0.0, 0.0, 0.0), lightContribution * spec) * shadingData.energyCompensation;

    // TODO:
    // - make sure that the shadow use a perspective projection
    let scatterVoH = saturate(dot(shadingData.V, -Ldirection));
    let forwardScatter = exp2(scatterVoH * mat.subsurfacePower - mat.subsurfacePower);
    let backScatter = saturate(shadingData.NoL * mat.subsurfaceThickness + (1.0 - mat.subsurfaceThickness)) * 0.5;
    let subsurface = mix(backScatter, 1.0, forwardScatter) * (1.0 - mat.subsurfaceThickness);
    shadingData.directDiffuse += mat.subsurfaceColor * (subsurface * diffuseLambertConstant());
}
