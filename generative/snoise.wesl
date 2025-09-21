import package::math::mod289::{mod289_2, mod289_3, mod289_4};
import package::math::permute::{permute, permute3, permute4};
import package::math::taylorInvSqrt::{taylorInvSqrt, taylorInvSqrt4};
import package::math::grad4::grad4;

/*
contributors: [Stefan Gustavson, Ian McEwan]
description: Simplex Noise https://github.com/stegu/webgl-noise
use: snoise2/3/4(<vec2f|vec3f|vec4f> pos)
license: |
    Copyright 2021-2023 by Stefan Gustavson and Ian McEwan.
    Published under the terms of the MIT license:
    https://opensource.org/license/mit/
*/

fn snoise2(v: vec2f) -> f32 {
    let C = vec4(0.211324865405187,  // (3.0-sqrt(3.0))/6.0
                 0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
                 -0.577350269189626, // -1.0 + 2.0 * C.x
                 0.024390243902439); // 1.0 / 41.0
    // First corner
    var i  = floor(v + dot(v, C.yy) );
    let x0 = v -   i + dot(i, C.xx);

    // Other corners
    //i1.x = step( x0.y, x0.x ); // x0.x > x0.y ? 1.0 : 0.0
    //i1.y = 1.0 - i1.x;
    let i1 = select(vec2(0.0, 1.0), vec2(1.0, 0.0), x0.x > x0.y);
    // x0 = x0 - 0.0 + 0.0 * C.xx ;
    // x1 = x0 - i1 + 1.0 * C.xx ;
    // x2 = x0 - 1.0 + 2.0 * C.xx ;
    let x12 = x0.xyxy + C.xxzz - vec4(i1, 0.0, 0.0);

    // Permutations
    i = mod289_2(i); // Avoid truncation effects in permutation
    let p = permute3( permute3( i.y + vec3(0.0, i1.y, 1.0 )) + i.x + vec3(0.0, i1.x, 1.0 ));

    var m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), vec3(0.0));
    m = m*m;
    m = m*m;

    // Gradients: 41 points uniformly over a line, mapped onto a diamond.
    // The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)

    let x = 2.0 * fract(p * C.www) - 1.0;
    let h = abs(x) - 0.5;
    let ox = floor(x + 0.5);
    let a0 = x - ox;

    // Normalise gradients implicitly by scaling m
    // Approximation of: m *= inversesqrt( a0*a0 + h*h );
    m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );

    // Compute final noise value at P
    let gx  = a0.x  * x0.x  + h.x  * x0.y;
    let gyz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, vec3(gx, gyz));
}

fn snoise3(v: vec3f) -> f32 {
    let C = vec2(1.0/6.0, 1.0/3.0) ;
    let D = vec4(0.0, 0.5, 1.0, 2.0);

    // First corner
    var i  = floor(v + dot(v, C.yyy) );
    let x0 =   v - i + dot(i, C.xxx) ;

    // Other corners
    let g = step(x0.yzx, x0.xyz);
    let l = 1.0 - g;
    let i1 = min( g.xyz, l.zxy );
    let i2 = max( g.xyz, l.zxy );

    //   x0 = x0 - 0.0 + 0.0 * C.xxx;
    //   x1 = x0 - i1  + 1.0 * C.xxx;
    //   x2 = x0 - i2  + 2.0 * C.xxx;
    //   x3 = x0 - 1.0 + 3.0 * C.xxx;
    let x1 = x0 - i1 + C.xxx;
    let x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
    let x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

    // Permutations
    i = mod289_3(i);
    let p = permute4( permute4( permute4(
                i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
            + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
            + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

    // Gradients: 7x7 points over a square, mapped onto an octahedron.
    // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
    let n_ = 0.142857142857; // 1.0/7.0
    let  ns = n_ * D.wyz - D.xzx;

    let j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

    let x_ = floor(j * ns.z);
    let y_ = floor(j - 7.0 * x_ );    // mod(j,N)

    let x = x_ *ns.x + ns.yyyy;
    let y = y_ *ns.x + ns.yyyy;
    let h = 1.0 - abs(x) - abs(y);

    let b0 = vec4( x.xy, y.xy );
    let b1 = vec4( x.zw, y.zw );

    //let s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
    //let s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
    let s0 = floor(b0)*2.0 + 1.0;
    let s1 = floor(b1)*2.0 + 1.0;
    let sh = -step(h, vec4(0.0));

    let a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
    let a1 = b1.xzyw + s1.xzyw*sh.zzww ;

    var p0 = vec3(a0.xy,h.x);
    var p1 = vec3(a0.zw,h.y);
    var p2 = vec3(a1.xy,h.z);
    var p3 = vec3(a1.zw,h.w);

    //Normalise gradients
    let norm = taylorInvSqrt4(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    // Mix final noise value
    var m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), vec4(0.0));
    m = m * m;
    return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1),
                                dot(p2,x2), dot(p3,x3) ) );
}

fn snoise4(v: vec4f) -> f32 {
    let C = vec4(0.138196601125011,   // (5 - sqrt(5))/20  G4
                 0.276393202250021,   // 2 * G4
                 0.414589803375032,   // 3 * G4
                 -0.447213595499958); // -1 + 4 * G4

    // First corner
    var i  = floor(v + dot(v, vec4(.309016994374947451)) ); // (sqrt(5) - 1)/4
    let x0 = v -   i + dot(i, C.xxxx);

    // Other corners

    // Rank sorting originally contributed by Bill Licea-Kane, AMD (formerly ATI)
    let isX = step( x0.yzw, x0.xxx );
    let isYZ = step( x0.zww, x0.yyz );
    //  i0.x = dot( isX, vec3( 1.0 ) );
    var i0 = vec4(isX.x + isX.y + isX.z, 1.0 - isX);
    //  i0.y += dot( isYZ.xy, vec2( 1.0 ) );
    i0 += vec4(0.0, isYZ.x + isYZ.y, 1.0 - isYZ.xy);
    i0 += vec4(0.0, 0.0, isYZ.z, 1.0 - isYZ.z);

    // i0 now contains the unique values 0,1,2,3 in each channel
    let i3 = clamp( i0, vec4(0.0), vec4(1.0) );
    let i2 = clamp( i0 - 1.0, vec4(0.0), vec4(1.0) );
    let i1 = clamp( i0 - 2.0, vec4(0.0), vec4(1.0) );

    //  x0 = x0 - 0.0 + 0.0 * C.xxxx
    //  x1 = x0 - i1  + 1.0 * C.xxxx
    //  x2 = x0 - i2  + 2.0 * C.xxxx
    //  x3 = x0 - i3  + 3.0 * C.xxxx
    //  x4 = x0 - 1.0 + 4.0 * C.xxxx
    let x1 = x0 - i1 + C.xxxx;
    let x2 = x0 - i2 + C.yyyy;
    let x3 = x0 - i3 + C.zzzz;
    let x4 = x0 + C.wwww;

    // Permutations
    i = mod289_4(i);
    let j0 = permute( permute( permute( permute(i.w) + i.z) + i.y) + i.x);
    let j1 = permute4( permute4( permute4( permute4 (
                i.w + vec4(i1.w, i2.w, i3.w, 1.0 ))
            + i.z + vec4(i1.z, i2.z, i3.z, 1.0 ))
            + i.y + vec4(i1.y, i2.y, i3.y, 1.0 ))
            + i.x + vec4(i1.x, i2.x, i3.x, 1.0 ));

    // Gradients: 7x7x6 points over a cube, mapped onto a 4-cross polytope
    // 7*7*6 = 294, which is close to the ring size 17*17 = 289.
    let ip = vec4(1.0/294.0, 1.0/49.0, 1.0/7.0, 0.0) ;

    var p0 = grad4(j0,   ip);
    var p1 = grad4(j1.x, ip);
    var p2 = grad4(j1.y, ip);
    var p3 = grad4(j1.z, ip);
    var p4 = grad4(j1.w, ip);

    // Normalise gradients
    let norm = taylorInvSqrt4(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;
    p4 *= taylorInvSqrt(dot(p4,p4));

    // Mix contributions from the five corners
    var m0 = max(0.6 - vec3(dot(x0,x0), dot(x1,x1), dot(x2,x2)), vec3(0.0));
    var m1 = max(0.6 - vec2(dot(x3,x3), dot(x4,x4)            ), vec2(0.0));
    m0 = m0 * m0;
    m1 = m1 * m1;
    return 49.0 * ( dot(m0*m0, vec3( dot( p0, x0 ), dot( p1, x1 ), dot( p2, x2 )))
                + dot(m1*m1, vec2( dot( p3, x3 ), dot( p4, x4 ) ) ) ) ;
}

fn snoise22(x: vec2f) -> vec2f {
    let s  = snoise2(vec2( x ));
    let s1 = snoise2(vec2( x.y - 19.1, x.x + 47.2 ));
    return vec2( s , s1 );
}

fn snoise33(x: vec3f) -> vec3f {
    let s  = snoise3(vec3( x ));
    let s1 = snoise3(vec3( x.y - 19.1 , x.z + 33.4 , x.x + 47.2 ));
    let s2 = snoise3(vec3( x.z + 74.2 , x.x - 124.5 , x.y + 99.4 ));
    return vec3( s , s1 , s2 );
}

fn snoise34(x: vec4f) -> vec3f {
    let s  = snoise4(vec4( x ));
    let s1 = snoise4(vec4( x.y - 19.1 , x.z + 33.4 , x.x + 47.2, x.w ));
    let s2 = snoise4(vec4( x.z + 74.2 , x.x - 124.5 , x.y + 99.4, x.w ));
    return vec3( s , s1 , s2 );
}
