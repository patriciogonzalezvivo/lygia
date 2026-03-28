/*
contributors: Shadi El Hajj
description: Light attenuation equation
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

fn attenuation(dist: f32) -> f32 {
    const LIGHT_ATTENUATION_CONSTANT: f32 = 0.0;
    const LIGHT_ATTENUATION_LINEAR: f32 = 0.0;
    const LIGHT_ATTENUATION_EXPONENTIAL: f32 = 1.0;
    const LIGHT_ATTENUATION_EXPONENT: f32 = 2.0;
    return 1.0 / (
        LIGHT_ATTENUATION_CONSTANT + 
        LIGHT_ATTENUATION_LINEAR * dist +
        LIGHT_ATTENUATION_EXPONENTIAL * pow(dist, LIGHT_ATTENUATION_EXPONENT)
        );
}
