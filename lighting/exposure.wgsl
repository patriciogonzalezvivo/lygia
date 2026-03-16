/*
contributors:  Shadi El Hajj
description: Calculate camera exposure
use: float exposure(float aperture, float shutterSpeed, float sensitivity)
license: MIT License (MIT) Copyright (c) 2024 Shadi EL Hajj
*/

fn exposure(aperture: f32, shutterSpeed: f32, sensitivity: f32) -> f32 {
    let ev100 = log2((aperture * aperture) / shutterSpeed * 100.0 / sensitivity);
    return 1.0 / (pow(2.0, ev100) * 1.2);
}
