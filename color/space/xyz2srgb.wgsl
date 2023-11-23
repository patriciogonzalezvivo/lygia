#include "rgb2srgb.wgsl"

fn xyz2srgb(xyz: vec3f ) -> vec3f {
    let D65_XYZ_RGB_0 = vec3f( 3.24306333, -1.53837619, -0.49893282);
    let D65_XYZ_RGB_1 = vec3f(-0.96896309,  1.87542451,  0.04154303);
    let D65_XYZ_RGB_2 = vec3f( 0.05568392, -0.20417438,  1.05799454);
    return rgb2srgb(vec3f(  dot(D65_XYZ_RGB_0, xyz), 
                                dot(D65_XYZ_RGB_1, xyz),
                                dot(D65_XYZ_RGB_2, xyz)));
}
