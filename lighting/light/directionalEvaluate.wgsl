#include "../specular.wgsl"
#include "../diffuse.wgsl"

/*
contributors: [Patricio Gonzalez Vivo, Shadi El Hajj]
description: Calculate directional light
use: <void> lightDirectionalEvaluate(<LightDirectional> L, <Material> mat, inout <ShadingData> shadingData)
options:
    - DIFFUSE_FNC: diffuseOrenNayar, diffuseBurley, diffuseLambert (default)
    - LIGHT_POSITION: in GlslViewer is u_light
    - LIGHT_DIRECTION
    - LIGHT_COLOR: in GlslViewer is u_lightColor
    - LIGHT_INTENSITY: in GlslViewer is u_lightIntensity
    - RAYMARCH_SHADOWS: enable raymarched shadows
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn lightDirectionalEvaluate(L: LightDirectional, mat: Material, shadingData: ShadingData) {

    shadingData.L = L.direction;
    shadingData.H = normalize(L.direction + shadingData.V);
    shadingData.NoL = saturate(dot(shadingData.N, L.direction));
    shadingData.NoH = saturate(dot(shadingData.N, shadingData.H));

    let shadow = raymarchSoftShadow(mat.position, L.direction);
    let shadow = 1.0;

    let dif = diffuse(shadingData);
    let spec = specular(shadingData);

    let lightContribution = L.color * L.intensity * shadow * shadingData.NoL;
    shadingData.directDiffuse  += max(vec3f(0.0, 0.0, 0.0), shadingData.diffuseColor * lightContribution * dif);
    shadingData.directSpecular += max(vec3f(0.0, 0.0, 0.0), lightContribution * spec) * shadingData.energyCompensation;

    let scatterVoH = saturate(dot(shadingData.V, -L.direction));
    let forwardScatter = exp2(scatterVoH * mat.subsurfacePower - mat.subsurfacePower);
    let backScatter = saturate(shadingData.NoL * mat.subsurfaceThickness + (1.0 - mat.subsurfaceThickness)) * 0.5;
    let subsurface = mix(backScatter, 1.0, forwardScatter) * (1.0 - mat.subsurfaceThickness);
    shadingData.directDiffuse += mat.subsurfaceColor * (subsurface * diffuseLambertConstant());
}
