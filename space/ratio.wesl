/*
contributors: Patricio Gonzalez Vivo
description: "Fix the aspect ratio of a space keeping things squared for you, \nin\
    \ a similar way that aspect.glsl does, but while scaling the \nspace to keep the\
    \ entire 0.0,0.0 ~ 1.0,1.0 range visible\n"
examples:
    - https://raw.githubusercontent.com/patriciogonzalezvivo/lygia_examples/main/draw_shapes.frag
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn ratio(st: vec2f, s: vec2f) -> vec2f {
    return mix( vec2f((st.x*s.x/s.y)-(s.x*.5-s.y*.5)/s.y,st.y),
                vec2f(st.x,st.y*(s.y/s.x)-(s.y*.5-s.x*.5)/s.x),
                step(s.x,s.y) );
}
