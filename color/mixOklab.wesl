import package::color::space::rgb2oklab::RGB2OKLAB_B;
import package::color::space::oklab2rgb::OKLAB2RGB_B;

/*
contributors:
    - Bjorn Ottosson
    - Inigo Quiles
description: |
    Mix function by Inigo Quiles (https://www.shadertoy.com/view/ttcyRS) 
    utilizing Bjorn Ottosso's OkLab color space, which is provide smooth stransitions 
    Learn more about it [his article](https://bottosson.github.io/posts/oklab/)
options:
    - MIXOKLAB_SRGB: by default colA and colB use linear RGB. If you want to use sRGB define this flag
examples:
    - /shaders/color_mix.frag
license: 
    - MIT License (MIT) Copyright (c) 2020 Björn Ottosson
    - MIT License (MIT) Copyright (c) 2020 Inigo Quilez
*/

fn mixOklab( colA: vec3f, colB: vec3f, h: f32 ) -> vec3f {
    
    // rgb to cone (arg of pow can't be negative)
    let lmsA = pow( RGB2OKLAB_B*colA, vec3f(0.33333) );
    let lmsB = pow( RGB2OKLAB_B*colB, vec3f(0.33333) );

    let lms = mix( lmsA, lmsB, h );

    // cone to rgb
    return OKLAB2RGB_B*(lms*lms*lms);
}
