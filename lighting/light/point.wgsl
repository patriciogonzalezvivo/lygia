/*
contributors: Patricio Gonzalez Vivo
description: Point light structure
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

struct LightPoint {
    var position: vec3f;
    var color: vec3f;
    var intensity: f32;
    var falloff: f32;
};
