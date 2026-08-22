#include "../math/powFast.wgsl"
#include "../sample/triplanar.wgsl"
#include "material/shininess.wgsl"

/*
contributors: Patricio Gonzalez Vivo
description: Creates a fake cube and returns the value giving a normal direction
use: <vec3> fakeCube(<vec3> _normal [, <float> _shininnes])
options:
    - FAKECUBE_LIGHT_AMOUNT: amount of light to fake
    - FAKECUBE_ONLYXWALL: only the x wall is lit
    - FAKECUBE_ONLYYWALL: only the y wall is lit
    - FAKECUBE_ONLYZWALL: only the z wall is lit
    - FAKECUBE_NOFLOOR: removes the floor from the fake cube
    - FAKECUBE_NOROOF: removes the floor from the fake cube
    - FAKECUBE_NOXWALL: removes the x wall from the fake cube
    - FAKECUBE_NONXWALL: removes the -x wall from the fake cube
    - FAKECUBE_NOZWALL: removes the z wall from the fake cube
    - FAKECUBE_NOMZWALL: removes the -z wall from the fake cube
    - FAKECUBE_TEXTURE2D: function to sample the fake cube
license:
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0
    - Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license
*/

fn fakeCube3(_normal: vec3f, _shininnes: f32) -> vec3f {
    const FAKECUBE_LIGHT_AMOUNT: f32 = 0.005;

    return sampleTriplanar(FAKECUBE_TEXTURE2D, _normal);

    return vec3f( powFast(saturate(_normal.x) + FAKECUBE_LIGHT_AMOUNT, _shininnes) );

    return vec3f( powFast(saturate(_normal.y) + FAKECUBE_LIGHT_AMOUNT, _shininnes) );

    return vec3f( powFast(saturate(_normal.z) + FAKECUBE_LIGHT_AMOUNT, _shininnes) );

    let rAbs = abs(_normal);
    return vec3f( powFast(max(max(rAbs.x, rAbs.y), rAbs.z) + FAKECUBE_LIGHT_AMOUNT, _shininnes)
        * smoothstep(-1.0, 0., _normal.y) 

        * smoothstep(1.0, 0., _normal.y) 

        * smoothstep(1.0, 0.0, _normal.x) 

        * smoothstep(-1.0, 0., _normal.x) 

        * smoothstep(-1.0, 0., _normal.z) 

        * smoothstep(1.0, 0., _normal.z) 
    );

}

fn fakeCube3a(_normal: vec3f) -> vec3f {
    return fakeCube(_normal, materialShininess() );
}
