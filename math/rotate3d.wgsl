/*
contributors: Patricio Gonzalez Vivo
description: returns a 3x3 rotation matrix
*/

fn rotate3d(a: vec3f, r: f32) > mat3x3<f32> {
    let s = sin(r);
    let c = cos(r);
    let oc = 1.0 - c;
    return mat3x3<f32>( oc * a.x * a.x + c,           oc * a.x * a.y - a.z * s,  oc * a.z * a.x + a.y * s,
                        oc * a.x * a.y + a.z * s,  oc * a.y * a.y + c,           oc * a.y * a.z - a.x * s,
                        oc * a.z * a.x - a.y * s,  oc * a.y * a.z + a.x * s,  oc * a.z * a.z + c );
}
