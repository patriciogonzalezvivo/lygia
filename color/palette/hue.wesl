import package::math::mod::mod3;

/*
contributors: Patricio Gonzalez Vivo
description: "Physical Hue. \n\nRatio: \n* 1/3 = neon\n* 1/4 = refracted\n* 1/5+ =\ approximate white\n"
examples:
    - https://raw.githubusercontent.com/eduardfossas/lygia-study-examples/main/color/palette/hue.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn hue(x: f32, r: f32) -> vec3f { 
    let v = abs( mod3(x + vec3f(0.0,1.0,2.0) * r, vec3f(1.0)) * 2.0 - 1.0);
    return v * v * (3.0 - 2.0 * v);
}
