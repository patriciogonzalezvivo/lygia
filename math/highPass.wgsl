/*
contributors: Patricio Gonzalez Vivo
description: bias high pass
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn highPass(v: f32, b: f32) -> f32 { return max(v - b, 0.0) / (1.0 - b); }