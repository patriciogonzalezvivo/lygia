import lygia::lighting::common::schlick::schlick;

/*
contributors: Patricio Gonzalez Vivo
description: Resolve fresnel coefficient
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn fresnel(f0: f32, NoV: f32) -> f32 {
    return schlick(f0, 1.0, NoV);
}
