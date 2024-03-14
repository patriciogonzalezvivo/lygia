/*
contributors: Patricio Gonzalez Vivo
description: bias high pass
*/

fn highPass(v: f32, b: f32) -> f32 { return max(v - b, 0.0) / (1.0 - b); }