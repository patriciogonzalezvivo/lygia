/*
contributors: Patricio Gonzalez Vivo
description: Directional Light Structure
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

struct LightDirectional {
    var direction: vec3f;
    var color: vec3f;
    var intensity: f32;
};
