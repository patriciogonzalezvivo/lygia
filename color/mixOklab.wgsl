#include "space/srgb2rgb.wgsl"
#include "space/rgb2srgb.wgsl"

#include "space/oklab2rgb.wgsl"
#include "space/rgb2oklab.wgsl"

fn mixOklab( colA: vec3f, colB: vec3f, h: f32 ) -> vec3f {
    
    // rgb to cone (arg of pow can't be negative)
    let lmsA = pow( RGB2OKLAB_B*colA, vec3f(0.33333) );
    let lmsB = pow( RGB2OKLAB_B*colB, vec3f(0.33333) );

    let lms = mix( lmsA, lmsB, h );

    // cone to rgb
    return OKLAB2RGB_B*(lms*lms*lms);
}
