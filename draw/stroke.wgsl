/*
contributors: Patricio Gonzalez Vivo
description: Fill a stroke in a SDF. From PixelSpiritDeck https://github.com/patriciogonzalezvivo/PixelSpiritDeck
use: stroke(<float> sdf, <float> size, <float> width [, <float> edge])
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn stroke(x: f32, size: f32, w: f32, edge: f32) -> f32 {
    return saturate(smoothstep(size - edge, size + edge, x + w * 0.5) - smoothstep(size - edge, size + edge, x - w * 0.5));
}