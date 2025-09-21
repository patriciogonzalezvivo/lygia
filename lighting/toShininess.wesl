/*
contributors: Patricio Gonzalez Vivo
description: Convertes from PBR roughness/metallic to a shininess factor (typaclly use on diffuse/specular/ambient workflow)
use: <f32> toShininess(<f32> roughness, <f32> metallic)
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn toShininess(roughness: f32, metallic: f32) -> f32 {
    var s = 0.95 - roughness * 0.5;
    s *= s;
    s *= s;
    return s * (80.0 + 160.0 * (1.0-metallic));
}
