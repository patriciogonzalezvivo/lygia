/*
contributors: Patricio Gonzalez Vivo
description: Fill a stroke in a SDF. From PixelSpiritDeck https://github.com/patriciogonzalezvivo/PixelSpiritDeck
use: stroke(<f32> sdf, <f32> size, <f32> width [, <f32> edge])
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

import lygia::math::aastep::aastep;
import lygia::math::saturate::saturate;

fn stroke(x: f32, size: f32, w: f32) -> f32 {
    let d = aastep(size, x + w * 0.5) - aastep(size, x - w * 0.5);
    return saturate(d);
}

fn strokeEdge(x: f32, size: f32, w: f32, edge: f32) -> f32 {
    let d = smoothstep(size - edge, size + edge, x + w * 0.5) - smoothstep(size - edge, size + edge, x - w * 0.5);
    return saturate(d);
}