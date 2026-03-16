#include "../toShininess.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Get material shininess property from GlslViewer's defines https://github.com/patriciogonzalezvivo/glslViewer/wiki/GlslViewer-DEFINES#material-defines
use: vec4 materialShininess()
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn materialShininess() -> f32 {
    let shininess = 15.0;

    shininess = MATERIAL_SHININESS;

    let roughness = materialRoughness();
    let metallic = materialMetallic();
    shininess = toShininess(roughness, metallic);

    return shininess;
}
