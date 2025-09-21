/*
contributors: Patricio Gonzalez Vivo
description: Heatmap palette
examples:
    - https://raw.githubusercontent.com/eduardfossas/lygia-study-examples/main/color/palette/heatmap.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn heatmap(v: f32) -> vec3f{
    let r = v * 2.1 - vec3f(1.8, 1.14, 0.3);
    return 1.0 - r * r;
}
