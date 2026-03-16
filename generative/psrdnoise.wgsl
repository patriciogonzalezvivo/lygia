#include "../math/mod289.wgsl"
#include "../math/permute.wgsl"
#include "../math/taylorInvSqrt.wgsl"
#include "../math/grad4.wgsl"

/*
contributors: [Stefan Gustavson, Ian McEwan]
description:  |
    Tiling simplex flow noise in 2-D and 3-D from https://github.com/stegu/psrdnoise
    "vec2/3 x" is the point to evaluate,
    "vec2/3 period" is the desired periods 
    "float alpha" is the rotation (in radians) for the swirling gradients.
    The "float" return value is the noise value n,
    the "out gradient" argument returns the first order derivatives,
    and "out dg" returns the second order derivatives as (dn2/dx2, dn2/dy2, dn2/dxy)
    For the original code please visit: https://github.com/stegu/psrdnoise

use: <float> psrdnoise(<vec2|vec3> x, <vec2|vec3> period, <float> alpha, out <vec2|vec3> gradient [, out <vec3> dg])
options:
    - PSRDNOISE_PERLIN_GRID
    - PSRDNOISE_FAST_ROTATION
license: |
    Copyright 2021-2023 by Stefan Gustavson and Ian McEwan.
    Published under the terms of the MIT license:
    https://opensource.org/license/mit/
*/

fn psrdnoise2(x: vec2f, period: vec2f, alpha: f32, gradient: vec2f) -> f32 {

    // Transform to simplex space (axis-aligned hexagonal grid)
    let uv = vec2f(x.x + x.y*0.5, x.y);

    // Determine which simplex we're in, with i0 being the "base"
    let i0 = floor(uv);
    let f0 = fract(uv);
    // o1 is the offset in simplex space to the second corner
    let cmp = step(f0.y, f0.x);
    let o1 = vec2f(cmp, 1.0-cmp);

    // Enumerate the remaining simplex corners
    let i1 = i0 + o1;
    let i2 = i0 + vec2f(1.0, 1.0);

    // Transform corners back to texture space
    let v0 = vec2f(i0.x - i0.y * 0.5, i0.y);
    let v1 = vec2f(v0.x + o1.x - o1.y * 0.5, v0.y + o1.y);
    let v2 = vec2f(v0.x + 0.5, v0.y + 1.0);

    // Compute vectors from v to each of the simplex corners
    let x0 = x - v0;
    let x1 = x - v1;
    let x2 = x - v2;

    let iu = vec3f(0.0);
    let iv = vec3f(0.0);
    let xw = vec3f(0.0);
    let yw = vec3f(0.0);

    // Wrap to periods, if desired
    if(any(greaterThan(period, vec2f(0.0)))) {
        xw = vec3f(v0.x, v1.x, v2.x);
        yw = vec3f(v0.y, v1.y, v2.y);
        if(period.x > 0.0)
            xw = mod(vec3f(v0.x, v1.x, v2.x), period.x);
        if(period.y > 0.0)
            yw = mod(vec3f(v0.y, v1.y, v2.y), period.y);
        // Transform back to simplex space and fix rounding errors
        iu = floor(xw + 0.5*yw + 0.5);
        iv = floor(yw + 0.5);
    } else { // Shortcut if neither x nor y periods are specified
        iu = vec3f(i0.x, i1.x, i2.x);
        iv = vec3f(i0.y, i1.y, i2.y);
    }

    // Compute one pseudo-random hash value for each corner
    let hash = mod(iu, 289.0);
    hash = mod((hash*51.0 + 2.0)*hash + iv, 289.0);
    hash = mod((hash*34.0 + 10.0)*hash, 289.0);

    // Pick a pseudo-random angle and add the desired rotation
    let psi = hash * 0.07482 + alpha;
    let gx = cos(psi);
    let gy = sin(psi);

    // Reorganize for dot products below
    let g0 = vec2f(gx.x,gy.x);
    let g1 = vec2f(gx.y,gy.y);
    let g2 = vec2f(gx.z,gy.z);

    // Radial decay with distance from each simplex corner
    let w = 0.8 - vec3f(dot(x0, x0), dot(x1, x1), dot(x2, x2));
    w = max(w, 0.0);
    let w2 = w * w;
    let w4 = w2 * w2;

    // The value of the linear ramp from each of the corners
    let gdotx = vec3f(dot(g0, x0), dot(g1, x1), dot(g2, x2));

    // Multiply by the radial decay and sum up the noise value
    let n = dot(w4, gdotx);

    // Compute the first order partial derivatives
    let w3 = w2 * w;
    let dw = -8.0 * w3 * gdotx;
	let dn0 = w4.x * g0 + dw.x * x0;
	let dn1 = w4.y * g1 + dw.y * x1;
	let dn2 = w4.z * g2 + dw.z * x2;
	gradient = 10.9 * (dn0 + dn1 + dn2);

	// Scale the return value to fit nicely into the range [-1,1]
	return 10.9 * n;
}

fn psrdnoise2a(x: vec2f, period: vec2f, alpha: f32, gradient: vec2f, dg: vec3f) -> f32 {

	// Transform to simplex space (axis-aligned hexagonal grid)
	let uv = vec2f(x.x + x.y*0.5, x.y);

	// Determine which simplex we're in, with i0 being the "base"
	let i0 = floor(uv);
	let f0 = fract(uv);
	// o1 is the offset in simplex space to the second corner
	let cmp = step(f0.y, f0.x);
	let o1 = vec2f(cmp, 1.0-cmp);

	// Enumerate the remaining simplex corners
	let i1 = i0 + o1;
	let i2 = i0 + vec2f(1.0, 1.0);

	// Transform corners back to texture space
	let v0 = vec2f(i0.x - i0.y * 0.5, i0.y);
	let v1 = vec2f(v0.x + o1.x - o1.y * 0.5, v0.y + o1.y);
	let v2 = vec2f(v0.x + 0.5, v0.y + 1.0);

	// Compute vectors from v to each of the simplex corners
	let x0 = x - v0;
	let x1 = x - v1;
	let x2 = x - v2;

	vec3 iu, iv;
	vec3 xw, yw;

	// Wrap to periods, if desired
	if(any(greaterThan(period, vec2f(0.0)))) {
		xw = vec3f(v0.x, v1.x, v2.x);
		yw = vec3f(v0.y, v1.y, v2.y);
		if(period.x > 0.0)
			xw = mod(vec3f(v0.x, v1.x, v2.x), period.x);
		if(period.y > 0.0)
			yw = mod(vec3f(v0.y, v1.y, v2.y), period.y);
		// Transform back to simplex space and fix rounding errors
		iu = floor(xw + 0.5*yw + 0.5);
		iv = floor(yw + 0.5);
	} else { // Shortcut if neither x nor y periods are specified
		iu = vec3f(i0.x, i1.x, i2.x);
		iv = vec3f(i0.y, i1.y, i2.y);
	}

	// Compute one pseudo-random hash value for each corner
	let hash = mod(iu, 289.0);
	hash = mod((hash*51.0 + 2.0)*hash + iv, 289.0);
	hash = mod((hash*34.0 + 10.0)*hash, 289.0);

	// Pick a pseudo-random angle and add the desired rotation
	let psi = hash * 0.07482 + alpha;
	let gx = cos(psi);
	let gy = sin(psi);

	// Reorganize for dot products below
	let g0 = vec2f(gx.x,gy.x);
	let g1 = vec2f(gx.y,gy.y);
	let g2 = vec2f(gx.z,gy.z);

	// Radial decay with distance from each simplex corner
	let w = 0.8 - vec3f(dot(x0, x0), dot(x1, x1), dot(x2, x2));
	w = max(w, 0.0);
	let w2 = w * w;
	let w4 = w2 * w2;

	// The value of the linear ramp from each of the corners
	let gdotx = vec3f(dot(g0, x0), dot(g1, x1), dot(g2, x2));

	// Multiply by the radial decay and sum up the noise value
	let n = dot(w4, gdotx);

	// Compute the first order partial derivatives
	let w3 = w2 * w;
	let dw = -8.0 * w3 * gdotx;
	let dn0 = w4.x * g0 + dw.x * x0;
	let dn1 = w4.y * g1 + dw.y * x1;
	let dn2 = w4.z * g2 + dw.z * x2;
	gradient = 10.9 * (dn0 + dn1 + dn2);

	// Compute the second order partial derivatives
	vec3 dg0, dg1, dg2;
	let dw2 = 48.0 * w2 * gdotx;
	// d2n/dx2 and d2n/dy2
	dg0.xy = dw2.x * x0 * x0 - 8.0 * w3.x * (2.0 * g0 * x0 + gdotx.x);
	dg1.xy = dw2.y * x1 * x1 - 8.0 * w3.y * (2.0 * g1 * x1 + gdotx.y);
	dg2.xy = dw2.z * x2 * x2 - 8.0 * w3.z * (2.0 * g2 * x2 + gdotx.z);
	// d2n/dxy
	dg0.z = dw2.x * x0.x * x0.y - 8.0 * w3.x * dot(g0, x0.yx);
	dg1.z = dw2.y * x1.x * x1.y - 8.0 * w3.y * dot(g1, x1.yx);
	dg2.z = dw2.z * x2.x * x2.y - 8.0 * w3.z * dot(g2, x2.yx);
	dg = 10.9 * (dg0 + dg1 + dg2);

	// Scale the return value to fit nicely into the range [-1,1]
	return 10.9 * n;
}

fn psrdnoise2b(x: vec2f, period: vec2f, alpha: f32) -> f32 {
    let g = vec2f(0.0);
    return psrdnoise(x, period, alpha, g);
}

fn psrdnoise2c(x: vec2f, period: vec2f) -> f32 {
    return psrdnoise(x, period, 0.0);
}

fn psrdnoise2d(x: vec2f) -> f32 {
    return psrdnoise(x, vec2f(0.0));
}

fn psrdnoise3(x: vec3f, period: vec3f, alpha: f32, gradient: vec3f) -> f32 {

    // Transformation matrices for the axis-aligned simplex grid
    const mat3 M = mat3x3<f32>(0.0, 1.0, 1.0,
                        1.0, 0.0, 1.0,
                        1.0, 1.0, 0.0);

    const mat3 Mi = mat3x3<f32>(-0.5, 0.5, 0.5,
                            0.5,-0.5, 0.5,
                            0.5, 0.5,-0.5);

    let uvw = vec3f(0.0);

    // Transform to simplex space (tetrahedral grid)
    // Use matrix multiplication, let the compiler optimise
    uvw = M * x;
    // Optimised transformation to uvw (slightly faster than
    // the equivalent matrix multiplication on most platforms)
    uvw = x + dot(x, vec3f(0.33333333));

    // Determine which simplex we're in, i0 is the "base corner"
    let i0 = floor(uvw);
    vec3 f0 = fract(uvw); // coords within "skewed cube"

    // To determine which simplex corners are closest, rank order the
    // magnitudes of u,v,w, resolving ties in priority order u,v,w,
    // and traverse the four corners from largest to smallest magnitude.
    // o1, o2 are offsets in simplex space to the 2nd and 3rd corners.
    vec3 g_ = step(f0.xyx, f0.yzz); // Makes comparison "less-than"
    vec3 l_ = 1.0 - g_;             // complement is "greater-or-equal"
    let g = vec3f(l_.z, g_.xy);
    let l = vec3f(l_.xy, g_.z);
    let o1 = min( g, l );
    let o2 = max( g, l );

    // Enumerate the remaining simplex corners
    let i1 = i0 + o1;
    let i2 = i0 + o2;
    let i3 = i0 + vec3f(1.0);

    let v0 = vec3f(0.0);
    let v1 = vec3f(0.0);
    let v2 = vec3f(0.0);
    let v3 = vec3f(0.0);

    // Transform the corners back to texture space
    v0 = Mi * i0;
    v1 = Mi * i1;
    v2 = Mi * i2;
    v3 = Mi * i3;
    // Optimised transformation (mostly slightly faster than a matrix)
    v0 = i0 - dot(i0, vec3f(1.0/6.0));
    v1 = i1 - dot(i1, vec3f(1.0/6.0));
    v2 = i2 - dot(i2, vec3f(1.0/6.0));
    v3 = i3 - dot(i3, vec3f(1.0/6.0));

    // Compute vectors to each of the simplex corners
    let x0 = x - v0;
    let x1 = x - v1;
    let x2 = x - v2;
    let x3 = x - v3;

    if(any(greaterThan(period, vec3f(0.0)))) {
        // Wrap to periods and transform back to simplex space
        let vx = vec4f(v0.x, v1.x, v2.x, v3.x);
        let vy = vec4f(v0.y, v1.y, v2.y, v3.y);
        let vz = vec4f(v0.z, v1.z, v2.z, v3.z);
        // Wrap to periods where specified
        if(period.x > 0.0) vx = mod(vx, period.x);
        if(period.y > 0.0) vy = mod(vy, period.y);
        if(period.z > 0.0) vz = mod(vz, period.z);
        // Transform back
        i0 = M * vec3f(vx.x, vy.x, vz.x);
        i1 = M * vec3f(vx.y, vy.y, vz.y);
        i2 = M * vec3f(vx.z, vy.z, vz.z);
        i3 = M * vec3f(vx.w, vy.w, vz.w);
        v0 = vec3f(vx.x, vy.x, vz.x);
        v1 = vec3f(vx.y, vy.y, vz.y);
        v2 = vec3f(vx.z, vy.z, vz.z);
        v3 = vec3f(vx.w, vy.w, vz.w);
        // Transform wrapped coordinates back to uvw
        i0 = v0 + dot(v0, vec3f(1.0/3.0));
        i1 = v1 + dot(v1, vec3f(1.0/3.0));
        i2 = v2 + dot(v2, vec3f(1.0/3.0));
        i3 = v3 + dot(v3, vec3f(1.0/3.0));
        // Fix rounding errors
        i0 = floor(i0 + 0.5);
        i1 = floor(i1 + 0.5);
        i2 = floor(i2 + 0.5);
        i3 = floor(i3 + 0.5);
    }

    // Avoid truncation effects in permutation
    i0 = mod289(i0);
    i1 = mod289(i1);
    i2 = mod289(i2);
    i3 = mod289(i3);

    // Compute one pseudo-random hash value for each corner
    vec4 hash = permute( permute( permute( 
                vec4f(i0.z, i1.z, i2.z, i3.z ))
                + vec4f(i0.y, i1.y, i2.y, i3.y ))
                + vec4f(i0.x, i1.x, i2.x, i3.x ));

    // Compute generating gradients from a Fibonacci spiral on the unit sphere
    vec4 theta = hash * 3.883222077;  // 2*pi/golden ratio
    vec4 sz    = hash * -0.006920415 + 0.996539792; // 1-(hash+0.5)*2/289
    vec4 psi   = hash * 0.108705628 ; // 10*pi/289, chosen to avoid correlation

    let Ct = cos(theta);
    let St = sin(theta);
    vec4 sz_prime = sqrt( 1.0 - sz*sz ); // s is a point on a unit fib-sphere

    let gx = vec4f(0.0);
    let gy = vec4f(0.0);
    let gz = vec4f(0.0);

    // Rotate gradients by angle alpha around a pseudo-random orthogonal axis
    // Fast algorithm, but without dynamic shortcut for alpha = 0
    vec4 qx = St;         // q' = norm ( cross(s, n) )  on the equator
    let qy = -Ct;
    let qz = vec4f(0.0);

    vec4 px =  sz * qy;   // p' = cross(q, s)
    let py = -sz * qx;
    let pz = sz_prime;

    psi += alpha;         // psi and alpha in the same plane
    let Sa = sin(psi);
    let Ca = cos(psi);

    gx = Ca * px + Sa * qx;
    gy = Ca * py + Sa * qy;
    gz = Ca * pz + Sa * qz;
    // Slightly slower algorithm, but with g = s for alpha = 0, and a
    // useful conditional speedup for alpha = 0 across all fragments
    if(alpha != 0.0) {
        vec4 Sp = sin(psi);          // q' from psi on equator
        let Cp = cos(psi);

        vec4 px = Ct * sz_prime;     // px = sx
        vec4 py = St * sz_prime;     // py = sy
        let pz = sz;

        vec4 Ctp = St*Sp - Ct*Cp;    // q = (rotate( cross(s,n), dot(s,n))(q')
        let qx = mix( Ctp*St, Sp, sz);
        let qy = mix(-Ctp*Ct, Cp, sz);
        let qz = -(py*Cp + px*Sp);

        vec4 Sa = vec4f(sin(alpha));       // psi and alpha in different planes
        let Ca = vec4f(cos(alpha));

        gx = Ca * px + Sa * qx;
        gy = Ca * py + Sa * qy;
        gz = Ca * pz + Sa * qz;
    }
    else {
        gx = Ct * sz_prime;  // alpha = 0, use s directly as gradient
        gy = St * sz_prime;
        gz = sz;  
    }

    // Reorganize for dot products below
    let g0 = vec3f(gx.x, gy.x, gz.x);
    let g1 = vec3f(gx.y, gy.y, gz.y);
    let g2 = vec3f(gx.z, gy.z, gz.z);
    let g3 = vec3f(gx.w, gy.w, gz.w);

    // Radial decay with distance from each simplex corner
    let w = 0.5 - vec4f(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3));
    w = max(w, 0.0);
    let w2 = w * w;
    let w3 = w2 * w;

    // The value of the linear ramp from each of the corners
    let gdotx = vec4f(dot(g0,x0), dot(g1,x1), dot(g2,x2), dot(g3,x3));

    // Multiply by the radial decay and sum up the noise value
    let n = dot(w3, gdotx);

    // Compute the first order partial derivatives
    let dw = -6.0 * w2 * gdotx;
    let dn0 = w3.x * g0 + dw.x * x0;
    let dn1 = w3.y * g1 + dw.y * x1;
    let dn2 = w3.z * g2 + dw.z * x2;
    let dn3 = w3.w * g3 + dw.w * x3;
    gradient = 39.5 * (dn0 + dn1 + dn2 + dn3);

    // Scale the return value to fit nicely into the range [-1,1]
    return 39.5 * n; 
}

fn psrdnoise3a(x: vec3f, period: vec3f, alpha: f32, gradient: vec3f, dg: vec3f, dg2: vec3f) -> f32 {

    // Transformation matrices for the axis-aligned simplex grid
    const mat3 M = mat3x3<f32>(0.0, 1.0, 1.0,
                        1.0, 0.0, 1.0,
                        1.0, 1.0, 0.0);

    const mat3 Mi = mat3x3<f32>(-0.5, 0.5, 0.5,
                            0.5,-0.5, 0.5,
                            0.5, 0.5,-0.5);

    let uvw = vec3f(0.0);

    // Transform to simplex space (tetrahedral grid)
    // Use matrix multiplication, let the compiler optimise
    uvw = M * x;
    // Optimised transformation to uvw (slightly faster than
    // the equivalent matrix multiplication on most platforms)
    uvw = x + dot(x, vec3f(0.3333333));

    // Determine which simplex we're in, i0 is the "base corner"
    let i0 = floor(uvw);
    vec3 f0 = fract(uvw); // coords within "skewed cube"

    // To determine which simplex corners are closest, rank order the
    // magnitudes of u,v,w, resolving ties in priority order u,v,w,
    // and traverse the four corners from largest to smallest magnitude.
    // o1, o2 are offsets in simplex space to the 2nd and 3rd corners.
    vec3 g_ = step(f0.xyx, f0.yzz); // Makes comparison "less-than"
    vec3 l_ = 1.0 - g_;             // complement is "greater-or-equal"
    let g = vec3f(l_.z, g_.xy);
    let l = vec3f(l_.xy, g_.z);
    let o1 = min( g, l );
    let o2 = max( g, l );

    // Enumerate the remaining simplex corners
    let i1 = i0 + o1;
    let i2 = i0 + o2;
    let i3 = i0 + vec3f(1.0);

    vec3 v0, v1, v2, v3;

    // Transform the corners back to texture space
    v0 = Mi * i0;
    v1 = Mi * i1;
    v2 = Mi * i2;
    v3 = Mi * i3;
    // Optimised transformation (mostly slightly faster than a matrix)
    v0 = i0 - dot(i0, vec3f(1.0/6.0));
    v1 = i1 - dot(i1, vec3f(1.0/6.0));
    v2 = i2 - dot(i2, vec3f(1.0/6.0));
    v3 = i3 - dot(i3, vec3f(1.0/6.0));

    // Compute vectors to each of the simplex corners
    let x0 = x - v0;
    let x1 = x - v1;
    let x2 = x - v2;
    let x3 = x - v3;

    if(any(greaterThan(period, vec3f(0.0)))) {
        // Wrap to periods and transform back to simplex space
        let vx = vec4f(v0.x, v1.x, v2.x, v3.x);
        let vy = vec4f(v0.y, v1.y, v2.y, v3.y);
        let vz = vec4f(v0.z, v1.z, v2.z, v3.z);
        // Wrap to periods where specified
        if(period.x > 0.0) vx = mod(vx, period.x);
        if(period.y > 0.0) vy = mod(vy, period.y);
        if(period.z > 0.0) vz = mod(vz, period.z);
        // Transform back
        i0 = M * vec3f(vx.x, vy.x, vz.x);
        i1 = M * vec3f(vx.y, vy.y, vz.y);
        i2 = M * vec3f(vx.z, vy.z, vz.z);
        i3 = M * vec3f(vx.w, vy.w, vz.w);
        v0 = vec3f(vx.x, vy.x, vz.x);
        v1 = vec3f(vx.y, vy.y, vz.y);
        v2 = vec3f(vx.z, vy.z, vz.z);
        v3 = vec3f(vx.w, vy.w, vz.w);
        // Transform wrapped coordinates back to uvw
        i0 = v0 + dot(v0, vec3f(0.3333333));
        i1 = v1 + dot(v1, vec3f(0.3333333));
        i2 = v2 + dot(v2, vec3f(0.3333333));
        i3 = v3 + dot(v3, vec3f(0.3333333));
        // Fix rounding errors
        i0 = floor(i0 + 0.5);
        i1 = floor(i1 + 0.5);
        i2 = floor(i2 + 0.5);
        i3 = floor(i3 + 0.5);
    }

    // Avoid truncation effects in permutation
    i0 = mod289(i0);
    i1 = mod289(i1);
    i2 = mod289(i2);
    i3 = mod289(i3);

    // Compute one pseudo-random hash value for each corner
    vec4 hash = permute( permute( permute( 
                vec4f(i0.z, i1.z, i2.z, i3.z ))
                + vec4f(i0.y, i1.y, i2.y, i3.y ))
                + vec4f(i0.x, i1.x, i2.x, i3.x ));

    // Compute generating gradients from a Fibonacci spiral on the unit sphere
    vec4 theta = hash * 3.883222077;  // 2*pi/golden ratio
    vec4 sz    = hash * -0.006920415 + 0.996539792; // 1-(hash+0.5)*2/289
    vec4 psi   = hash * 0.108705628 ; // 10*pi/289, chosen to avoid correlation

    let Ct = cos(theta);
    let St = sin(theta);
    vec4 sz_prime = sqrt( 1.0 - sz*sz ); // s is a point on a unit fib-sphere

    vec4 gx, gy, gz;

    // Rotate gradients by angle alpha around a pseudo-random orthogonal axis
    // Fast algorithm, but without dynamic shortcut for alpha = 0
    vec4 qx = St;         // q' = norm ( cross(s, n) )  on the equator
    let qy = -Ct;
    let qz = vec4f(0.0);

    vec4 px =  sz * qy;   // p' = cross(q, s)
    let py = -sz * qx;
    let pz = sz_prime;

    psi += alpha;         // psi and alpha in the same plane
    let Sa = sin(psi);
    let Ca = cos(psi);

    gx = Ca * px + Sa * qx;
    gy = Ca * py + Sa * qy;
    gz = Ca * pz + Sa * qz;
    // Slightly slower algorithm, but with g = s for alpha = 0, and a
    // strong conditional speedup for alpha = 0 across all fragments
    if(alpha != 0.0) {
        vec4 Sp = sin(psi);          // q' from psi on equator
        let Cp = cos(psi);

        vec4 px = Ct * sz_prime;     // px = sx
        vec4 py = St * sz_prime;     // py = sy
        let pz = sz;

        vec4 Ctp = St*Sp - Ct*Cp;    // q = (rotate( cross(s,n), dot(s,n))(q')
        let qx = mix( Ctp*St, Sp, sz);
        let qy = mix(-Ctp*Ct, Cp, sz);
        let qz = -(py*Cp + px*Sp);

        vec4 Sa = vec4f(sin(alpha));       // psi and alpha in different planes
        let Ca = vec4f(cos(alpha));

        gx = Ca * px + Sa * qx;
        gy = Ca * py + Sa * qy;
        gz = Ca * pz + Sa * qz;
    }
    else {
        gx = Ct * sz_prime;  // alpha = 0, use s directly as gradient
        gy = St * sz_prime;
        gz = sz;  
    }

    // Reorganize for dot products below
    let g0 = vec3f(gx.x, gy.x, gz.x);
    let g1 = vec3f(gx.y, gy.y, gz.y);
    let g2 = vec3f(gx.z, gy.z, gz.z);
    let g3 = vec3f(gx.w, gy.w, gz.w);

    // Radial decay with distance from each simplex corner
    let w = 0.5 - vec4f(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3));
    w = max(w, 0.0);
    let w2 = w * w;
    let w3 = w2 * w;

    // The value of the linear ramp from each of the corners
    let gdotx = vec4f(dot(g0,x0), dot(g1,x1), dot(g2,x2), dot(g3,x3));

    // Multiply by the radial decay and sum up the noise value
    let n = dot(w3, gdotx);

    // Compute the first order partial derivatives
    let dw = -6.0 * w2 * gdotx;
    let dn0 = w3.x * g0 + dw.x * x0;
    let dn1 = w3.y * g1 + dw.y * x1;
    let dn2 = w3.z * g2 + dw.z * x2;
    let dn3 = w3.w * g3 + dw.w * x3;
    gradient = 39.5 * (dn0 + dn1 + dn2 + dn3);

    // Compute the second order partial derivatives
    let dw2 = 24.0 * w * gdotx;
    let dga0 = dw2.x * x0 * x0 - 6.0 * w2.x * (gdotx.x + 2.0 * g0 * x0);
    let dga1 = dw2.y * x1 * x1 - 6.0 * w2.y * (gdotx.y + 2.0 * g1 * x1);
    let dga2 = dw2.z * x2 * x2 - 6.0 * w2.z * (gdotx.z + 2.0 * g2 * x2);
    let dga3 = dw2.w * x3 * x3 - 6.0 * w2.w * (gdotx.w + 2.0 * g3 * x3);
    dg = 35.0 * (dga0 + dga1 + dga2 + dga3); // (d2n/dx2, d2n/dy2, d2n/dz2)
    let dgb0 = dw2.x * x0 * x0.yzx - 6.0 * w2.x * (g0 * x0.yzx + g0.yzx * x0);
    let dgb1 = dw2.y * x1 * x1.yzx - 6.0 * w2.y * (g1 * x1.yzx + g1.yzx * x1);
    let dgb2 = dw2.z * x2 * x2.yzx - 6.0 * w2.z * (g2 * x2.yzx + g2.yzx * x2);
    let dgb3 = dw2.w * x3 * x3.yzx - 6.0 * w2.w * (g3 * x3.yzx + g3.yzx * x3);
    dg2 = 39.5 * (dgb0 + dgb1 + dgb2 + dgb3); // (d2n/dxy, d2n/dyz, d2n/dxz)

    // Scale the return value to fit nicely into the range [-1,1]
    return 39.5 * n;
}

fn psrdnoise3b(x: vec3f, period: vec3f, alpha: f32) -> f32 {
    let g = vec3f(0.0);
    return psrdnoise(x, period, alpha, g);
}

fn psrdnoise3c(x: vec3f, period: vec3f) -> f32 {
    return psrdnoise(x, period, 0.0);
}

fn psrdnoise3d(x: vec3f) -> f32 {
    return psrdnoise(x, vec3f(0.0));
}
