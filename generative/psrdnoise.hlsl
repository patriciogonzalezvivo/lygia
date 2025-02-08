#include "../math/mod289.hlsl"
#include "../math/permute.hlsl"
#include "../math/taylorInvSqrt.hlsl"
#include "../math/grad4.hlsl"
#include "../math/mod.hlsl"
#include "../math/greaterThan.hlsl"

/*
contributors: [Stefan Gustavson, Ian McEwan]
description:  |
    Tiling simplex flow noise in 2-D and 3-D from https://github.com/stegu/psrdnoise
    "float2/3 x" is the point to evaluate,
    "float2/3 period" is the desired periods 
    "float alpha" is the rotation (in radians) for the swirling gradients.
    The "float" return value is the noise value n,
    the "out gradient" argument returns the first order derivatives,
    and "out dg" returns the second order derivatives as (dn2/dx2, dn2/dy2, dn2/dxy)
use: <float> psrdnoise(<float2|float3> x, <float2|float3> period, <float> alpha, out <float2|float3> gradient [, out <float3> dg])
options:
    - PSRDNOISE_PERLIN_GRID
    - PSRDNOISE_FAST_ROTATION
license: |
    Copyright 2021-2023 by Stefan Gustavson and Ian McEwan.
    Published under the terms of the MIT license:
    https://opensource.org/license/mit/
*/

#ifndef FNC_PSRFNOISE
#define FNC_PSRFNOISE

float psrdnoise(float2 x, float2 period, float alpha, out float2 gradient) {

	// Transform to simplex space (axis-aligned hexagonal grid)
	float2 uv = float2(x.x + x.y*0.5, x.y);

	// Determine which simplex we're in, with i0 being the "base"
	float2 i0 = floor(uv);
	float2 f0 = frac(uv);
	// o1 is the offset in simplex space to the second corner
	float cmp = step(f0.y, f0.x);
	float2 o1 = float2(cmp, 1.0-cmp);

	// Enumerate the remaining simplex corners
	float2 i1 = i0 + o1;
	float2 i2 = i0 + float2(1.0, 1.0);

	// Transform corners back to texture space
	float2 v0 = float2(i0.x - i0.y * 0.5, i0.y);
	float2 v1 = float2(v0.x + o1.x - o1.y * 0.5, v0.y + o1.y);
	float2 v2 = float2(v0.x + 0.5, v0.y + 1.0);

	// Compute vectors from v to each of the simplex corners
	float2 x0 = x - v0;
	float2 x1 = x - v1;
	float2 x2 = x - v2;

	float3 iu = float3(0.0, 0.0, 0.0);
    float3 iv = float3(0.0, 0.0, 0.0);
	float3 xw = float3(0.0, 0.0, 0.0);
    float3 yw = float3(0.0, 0.0, 0.0);

	// Wrap to periods, if desired
	if( any(greaterThan(period, float2(0.0, 0.0))) ) {
		xw = float3(v0.x, v1.x, v2.x);
		yw = float3(v0.y, v1.y, v2.y);
		if(period.x > 0.0)
			xw = mod(float3(v0.x, v1.x, v2.x), period.x);
		if(period.y > 0.0)
			yw = mod(float3(v0.y, v1.y, v2.y), period.y);
		// Transform back to simplex space and fix rounding errors
		iu = floor(xw + 0.5*yw + 0.5);
		iv = floor(yw + 0.5);
	} else { // Shortcut if neither x nor y periods are specified
		iu = float3(i0.x, i1.x, i2.x);
		iv = float3(i0.y, i1.y, i2.y);
	}

	// Compute one pseudo-random hash value for each corner
	float3 hash = mod(iu, 289.0);
	hash = mod((hash*51.0 + 2.0)*hash + iv, 289.0);
	hash = mod((hash*34.0 + 10.0)*hash, 289.0);

	// Pick a pseudo-random angle and add the desired rotation
	float3 psi = hash * 0.07482 + alpha;
	float3 gx = cos(psi);
	float3 gy = sin(psi);

	// Reorganize for dot products below
	float2 g0 = float2(gx.x,gy.x);
	float2 g1 = float2(gx.y,gy.y);
	float2 g2 = float2(gx.z,gy.z);

	// Radial decay with distance from each simplex corner
	float3 w = 0.8 - float3(dot(x0, x0), dot(x1, x1), dot(x2, x2));
	w = max(w, 0.0);
	float3 w2 = w * w;
	float3 w4 = w2 * w2;

	// The value of the linear ramp from each of the corners
	float3 gdotx = float3(dot(g0, x0), dot(g1, x1), dot(g2, x2));

	// Multiply by the radial decay and sum up the noise value
	float n = dot(w4, gdotx);

	// Compute the first order partial derivatives
	float3 w3 = w2 * w;
	float3 dw = -8.0 * w3 * gdotx;
	float2 dn0 = w4.x * g0 + dw.x * x0;
	float2 dn1 = w4.y * g1 + dw.y * x1;
	float2 dn2 = w4.z * g2 + dw.z * x2;
	gradient = 10.9 * (dn0 + dn1 + dn2);

	// Scale the return value to fit nicely into the range [-1,1]
	return 10.9 * n;
}

float psrdnoise(float2 x, float2 period, float alpha, out float2 gradient, out float3 dg) {

	// Transform to simplex space (axis-aligned hexagonal grid)
	float2 uv = float2(x.x + x.y*0.5, x.y);

	// Determine which simplex we're in, with i0 being the "base"
	float2 i0 = floor(uv);
	float2 f0 = frac(uv);
	// o1 is the offset in simplex space to the second corner
	float cmp = step(f0.y, f0.x);
	float2 o1 = float2(cmp, 1.0-cmp);

	// Enumerate the remaining simplex corners
	float2 i1 = i0 + o1;
	float2 i2 = i0 + float2(1.0, 1.0);

	// Transform corners back to texture space
	float2 v0 = float2(i0.x - i0.y * 0.5, i0.y);
	float2 v1 = float2(v0.x + o1.x - o1.y * 0.5, v0.y + o1.y);
	float2 v2 = float2(v0.x + 0.5, v0.y + 1.0);

	// Compute vectors from v to each of the simplex corners
	float2 x0 = x - v0;
	float2 x1 = x - v1;
	float2 x2 = x - v2;

	float3 iu, iv;
	float3 xw, yw;

	// Wrap to periods, if desired
	if(any(greaterThan(period, float2(0.0, 0.0)))) {
		xw = float3(v0.x, v1.x, v2.x);
		yw = float3(v0.y, v1.y, v2.y);
		if(period.x > 0.0)
			xw = mod(float3(v0.x, v1.x, v2.x), period.x);
		if(period.y > 0.0)
			yw = mod(float3(v0.y, v1.y, v2.y), period.y);
		// Transform back to simplex space and fix rounding errors
		iu = floor(xw + 0.5*yw + 0.5);
		iv = floor(yw + 0.5);
	} else { // Shortcut if neither x nor y periods are specified
		iu = float3(i0.x, i1.x, i2.x);
		iv = float3(i0.y, i1.y, i2.y);
	}

	// Compute one pseudo-random hash value for each corner
	float3 hash = mod(iu, 289.0);
	hash = mod((hash*51.0 + 2.0)*hash + iv, 289.0);
	hash = mod((hash*34.0 + 10.0)*hash, 289.0);

	// Pick a pseudo-random angle and add the desired rotation
	float3 psi = hash * 0.07482 + alpha;
	float3 gx = cos(psi);
	float3 gy = sin(psi);

	// Reorganize for dot products below
	float2 g0 = float2(gx.x,gy.x);
	float2 g1 = float2(gx.y,gy.y);
	float2 g2 = float2(gx.z,gy.z);

	// Radial decay with distance from each simplex corner
	float3 w = 0.8 - float3(dot(x0, x0), dot(x1, x1), dot(x2, x2));
	w = max(w, 0.0);
	float3 w2 = w * w;
	float3 w4 = w2 * w2;

	// The value of the linear ramp from each of the corners
	float3 gdotx = float3(dot(g0, x0), dot(g1, x1), dot(g2, x2));

	// Multiply by the radial decay and sum up the noise value
	float n = dot(w4, gdotx);

	// Compute the first order partial derivatives
	float3 w3 = w2 * w;
	float3 dw = -8.0 * w3 * gdotx;
	float2 dn0 = w4.x * g0 + dw.x * x0;
	float2 dn1 = w4.y * g1 + dw.y * x1;
	float2 dn2 = w4.z * g2 + dw.z * x2;
	gradient = 10.9 * (dn0 + dn1 + dn2);

	// Compute the second order partial derivatives
	float3 dg0, dg1, dg2;
	float3 dw2 = 48.0 * w2 * gdotx;
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

float psrdnoise(float2 x, float2 period, float alpha) {
    float2 g = float2(0.0, 0.0);
    return psrdnoise(x, period, alpha, g);
}

float psrdnoise(float2 x, float2 period) {
    return psrdnoise(x, period, 0.0);
}

float psrdnoise(float2 x) {
    return psrdnoise(x, float2(0.0, 0.0));
}

float psrdnoise(float3 x, float3 period, float alpha, out float3 gradient) {

#ifndef PSRDNOISE_PERLIN_GRID
    // Transformation matrices for the axis-aligned simplex grid
    static const float3x3 M = float3x3(0.0, 1.0, 1.0,
                                1.0, 0.0, 1.0,
                                1.0, 1.0, 0.0);

    static const float3x3 Mi = float3x3(-0.5, 0.5, 0.5,
                                0.5,-0.5, 0.5,
                                0.5, 0.5,-0.5);
#endif

    float3 uvw = float3(0.0, 0.0, 0.0);

    // Transform to simplex space (tetrahedral grid)
#ifndef PSRDNOISE_PERLIN_GRID
    // Use matrix multiplication, let the compiler optimise
    uvw = mul(M, x);
 #else
    // Optimised transformation to uvw (slightly faster than
    // the equivalent matrix multiplication on most platforms)
    uvw = x + dot(x, float3(0.33333333, 0.33333333, 0.33333333));
 #endif

    // Determine which simplex we're in, i0 is the "base corner"
    float3 i0 = floor(uvw);
    float3 f0 = frac(uvw); // coords within "skewed cube"

    // To determine which simplex corners are closest, rank order the
    // magnitudes of u,v,w, resolving ties in priority order u,v,w,
    // and traverse the four corners from largest to smallest magnitude.
    // o1, o2 are offsets in simplex space to the 2nd and 3rd corners.
    float3 g_ = step(f0.xyx, f0.yzz); // Makes comparison "less-than"
    float3 l_ = 1.0 - g_;             // complement is "greater-or-equal"
    float3 g = float3(l_.z, g_.xy);
    float3 l = float3(l_.xy, g_.z);
    float3 o1 = min( g, l );
    float3 o2 = max( g, l );

    // Enumerate the remaining simplex corners
    float3 i1 = i0 + o1;
    float3 i2 = i0 + o2;
    float3 i3 = i0 + float3(1.0, 1.0, 1.0);

    float3 v0 = float3(0.0, 0.0, 0.0);
    float3 v1 = float3(0.0, 0.0, 0.0);
    float3 v2 = float3(0.0, 0.0, 0.0);
    float3 v3 = float3(0.0, 0.0, 0.0);

    // Transform the corners back to texture space
#ifndef PSRDNOISE_PERLIN_GRID
    v0 = mul(Mi, i0);
    v1 = mul(Mi, i1);
    v2 = mul(Mi, i2);
    v3 = mul(Mi, i3);
#else
    // Optimised transformation (mostly slightly faster than a matrix)
    v0 = i0 - dot(i0, float3(1.0/6.0));
    v1 = i1 - dot(i1, float3(1.0/6.0));
    v2 = i2 - dot(i2, float3(1.0/6.0));
    v3 = i3 - dot(i3, float3(1.0/6.0));
#endif

    // Compute vectors to each of the simplex corners
    float3 x0 = x - v0;
    float3 x1 = x - v1;
    float3 x2 = x - v2;
    float3 x3 = x - v3;

    if(any(greaterThan(period, float3(0.0, 0.0, 0.0)))) {
        // Wrap to periods and transform back to simplex space
        float4 vx = float4(v0.x, v1.x, v2.x, v3.x);
        float4 vy = float4(v0.y, v1.y, v2.y, v3.y);
        float4 vz = float4(v0.z, v1.z, v2.z, v3.z);
        // Wrap to periods where specified
        if(period.x > 0.0) vx = mod(vx, period.x);
        if(period.y > 0.0) vy = mod(vy, period.y);
        if(period.z > 0.0) vz = mod(vz, period.z);
        // Transform back
#ifndef PSRDNOISE_PERLIN_GRID
        i0 = mul(M, float3(vx.x, vy.x, vz.x));
        i1 = mul(M, float3(vx.y, vy.y, vz.y));
        i2 = mul(M, float3(vx.z, vy.z, vz.z));
        i3 = mul(M, float3(vx.w, vy.w, vz.w));
#else
        v0 = float3(vx.x, vy.x, vz.x);
        v1 = float3(vx.y, vy.y, vz.y);
        v2 = float3(vx.z, vy.z, vz.z);
        v3 = float3(vx.w, vy.w, vz.w);
        // Transform wrapped coordinates back to uvw
        i0 = v0 + dot(v0, float3(1.0/3.0));
        i1 = v1 + dot(v1, float3(1.0/3.0));
        i2 = v2 + dot(v2, float3(1.0/3.0));
        i3 = v3 + dot(v3, float3(1.0/3.0));
#endif
        // Fix rounding errors
        i0 = floor(i0 + 0.5);
        i1 = floor(i1 + 0.5);
        i2 = floor(i2 + 0.5);
        i3 = floor(i3 + 0.5);
    }

    // Compute one pseudo-random hash value for each corner
    float4 hash = permute( permute( permute( 
                float4(i0.z, i1.z, i2.z, i3.z ))
                + float4(i0.y, i1.y, i2.y, i3.y ))
                + float4(i0.x, i1.x, i2.x, i3.x ));

    // Compute generating gradients from a Fibonacci spiral on the unit sphere
    float4 theta = hash * 3.883222077;  // 2*pi/golden ratio
    float4 sz    = hash * -0.006920415 + 0.996539792; // 1-(hash+0.5)*2/289
    float4 psi   = hash * 0.108705628 ; // 10*pi/289, chosen to avoid correlation

    float4 Ct = cos(theta);
    float4 St = sin(theta);
    float4 sz_prime = sqrt( 1.0 - sz*sz ); // s is a point on a unit fib-sphere

    float4 gx = float4(0.0, 0.0, 0.0, 0.0);
    float4 gy = float4(0.0, 0.0, 0.0, 0.0);
    float4 gz = float4(0.0, 0.0, 0.0, 0.0);

    // Rotate gradients by angle alpha around a pseudo-random orthogonal axis
#ifdef PSRDNOISE_FAST_ROTATION
    // Fast algorithm, but without dynamic shortcut for alpha = 0
    float4 qx = St;         // q' = norm ( cross(s, n) )  on the equator
    float4 qy = -Ct; 
    float4 qz = float4(0.0, 0.0, 0.0, 0.0);

    float4 px =  sz * qy;   // p' = cross(q, s)
    float4 py = -sz * qx;
    float4 pz = sz_prime;

    psi += alpha;         // psi and alpha in the same plane
    float4 Sa = sin(psi);
    float4 Ca = cos(psi);

    gx = Ca * px + Sa * qx;
    gy = Ca * py + Sa * qy;
    gz = Ca * pz + Sa * qz;
#else
    // Slightly slower algorithm, but with g = s for alpha = 0, and a
    // useful conditional speedup for alpha = 0 across all fragments
    if(alpha != 0.0) {
        float4 Sp = sin(psi);          // q' from psi on equator
        float4 Cp = cos(psi);

        float4 px = Ct * sz_prime;     // px = sx
        float4 py = St * sz_prime;     // py = sy
        float4 pz = sz;

        float4 Ctp = St*Sp - Ct*Cp;    // q = (rotate( cross(s,n), dot(s,n))(q')
        float4 qx = lerp( Ctp*St, Sp, sz);
        float4 qy = lerp(-Ctp*Ct, Cp, sz);
        float4 qz = -(py*Cp + px*Sp);

        float sin_a = sin(alpha);
        float cos_a = cos(alpha);
        float4 Sa = float4(sin_a, sin_a, sin_a, sin_a);       // psi and alpha in different planes
        float4 Ca = float4(cos_a, cos_a, cos_a, cos_a);

        gx = Ca * px + Sa * qx;
        gy = Ca * py + Sa * qy;
        gz = Ca * pz + Sa * qz;
    }
    else {
        gx = Ct * sz_prime;  // alpha = 0, use s directly as gradient
        gy = St * sz_prime;
        gz = sz;  
    }
#endif

    // Reorganize for dot products below
    float3 g0 = float3(gx.x, gy.x, gz.x);
    float3 g1 = float3(gx.y, gy.y, gz.y);
    float3 g2 = float3(gx.z, gy.z, gz.z);
    float3 g3 = float3(gx.w, gy.w, gz.w);

    // Radial decay with distance from each simplex corner
    float4 w = 0.5 - float4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3));
    w = max(w, 0.0);
    float4 w2 = w * w;
    float4 w3 = w2 * w;

    // The value of the linear ramp from each of the corners
    float4 gdotx = float4(dot(g0,x0), dot(g1,x1), dot(g2,x2), dot(g3,x3));

    // Multiply by the radial decay and sum up the noise value
    float n = dot(w3, gdotx);

    // Compute the first order partial derivatives
    float4 dw = -6.0 * w2 * gdotx;
    float3 dn0 = w3.x * g0 + dw.x * x0;
    float3 dn1 = w3.y * g1 + dw.y * x1;
    float3 dn2 = w3.z * g2 + dw.z * x2;
    float3 dn3 = w3.w * g3 + dw.w * x3;
    gradient = 39.5 * (dn0 + dn1 + dn2 + dn3);

    // Scale the return value to fit nicely into the range [-1,1]
    return 39.5 * n;
  
}

float psrdnoise(float3 x, float3 period, float alpha, out float3 gradient, out float3 dg, out float3 dg2) {

#ifndef PSRDNOISE_PERLIN_GRID
    // Transformation matrices for the axis-aligned simplex grid
    static const float3x3 M = float3x3(0.0, 1.0, 1.0,
                        1.0, 0.0, 1.0,
                        1.0, 1.0, 0.0);

    static const float3x3 Mi = float3x3(-0.5, 0.5, 0.5,
                            0.5,-0.5, 0.5,
                            0.5, 0.5,-0.5);
#endif

    float3 uvw = float3(0.0, 0.0, 0.0);

    // Transform to simplex space (tetrahedral grid)
#ifndef PSRDNOISE_PERLIN_GRID
    // Use matrix multiplication, let the compiler optimise
    uvw = mul(M, x);
#else
    // Optimised transformation to uvw (slightly faster than
    // the equivalent matrix multiplication on most platforms)
    uvw = x + dot(x, float3(0.3333333, 0.3333333, 0.3333333));
#endif

    // Determine which simplex we're in, i0 is the "base corner"
    float3 i0 = floor(uvw);
    float3 f0 = frac(uvw); // coords within "skewed cube"

    // To determine which simplex corners are closest, rank order the
    // magnitudes of u,v,w, resolving ties in priority order u,v,w,
    // and traverse the four corners from largest to smallest magnitude.
    // o1, o2 are offsets in simplex space to the 2nd and 3rd corners.
    float3 g_ = step(f0.xyx, f0.yzz); // Makes comparison "less-than"
    float3 l_ = 1.0 - g_;             // complement is "greater-or-equal"
    float3 g = float3(l_.z, g_.xy);
    float3 l = float3(l_.xy, g_.z);
    float3 o1 = min( g, l );
    float3 o2 = max( g, l );

    // Enumerate the remaining simplex corners
    float3 i1 = i0 + o1;
    float3 i2 = i0 + o2;
    float3 i3 = i0 + float3(1.0, 1.0, 1.0);

    float3 v0, v1, v2, v3;

    // Transform the corners back to texture space
#ifndef PSRDNOISE_PERLIN_GRID
    v0 = mul(Mi, i0);
    v1 = mul(Mi, i1);
    v2 = mul(Mi, i2);
    v3 = mul(Mi, i3);
#else
    // Optimised transformation (mostly slightly faster than a matrix)
    v0 = i0 - dot(i0, float3(0.166666667, 0.166666667, 0.166666667));
    v1 = i1 - dot(i1, float3(0.166666667, 0.166666667, 0.166666667));
    v2 = i2 - dot(i2, float3(0.166666667, 0.166666667, 0.166666667));
    v3 = i3 - dot(i3, float3(0.166666667, 0.166666667, 0.166666667));
#endif

    // Compute vectors to each of the simplex corners
    float3 x0 = x - v0;
    float3 x1 = x - v1;
    float3 x2 = x - v2;
    float3 x3 = x - v3;

    if(any(greaterThan(period, float3(0.0, 0.0, 0.0)))) {
        // Wrap to periods and transform back to simplex space
        float4 vx = float4(v0.x, v1.x, v2.x, v3.x);
        float4 vy = float4(v0.y, v1.y, v2.y, v3.y);
        float4 vz = float4(v0.z, v1.z, v2.z, v3.z);
        // Wrap to periods where specified
        if(period.x > 0.0) vx = mod(vx, period.x);
        if(period.y > 0.0) vy = mod(vy, period.y);
        if(period.z > 0.0) vz = mod(vz, period.z);
        // Transform back
#ifndef PSRDNOISE_PERLIN_GRID
        i0 = mul(M, float3(vx.x, vy.x, vz.x));
        i1 = mul(M, float3(vx.y, vy.y, vz.y));
        i2 = mul(M, float3(vx.z, vy.z, vz.z));
        i3 = mul(M, float3(vx.w, vy.w, vz.w));
#else
        v0 = float3(vx.x, vy.x, vz.x);
        v1 = float3(vx.y, vy.y, vz.y);
        v2 = float3(vx.z, vy.z, vz.z);
        v3 = float3(vx.w, vy.w, vz.w);
        // Transform wrapped coordinates back to uvw
        i0 = v0 + dot(v0, float3(0.3333333, 0.3333333, 0.3333333));
        i1 = v1 + dot(v1, float3(0.3333333, 0.3333333, 0.3333333));
        i2 = v2 + dot(v2, float3(0.3333333, 0.3333333, 0.3333333));
        i3 = v3 + dot(v3, float3(0.3333333, 0.3333333, 0.3333333));
#endif
        // Fix rounding errors
        i0 = floor(i0 + 0.5);
        i1 = floor(i1 + 0.5);
        i2 = floor(i2 + 0.5);
        i3 = floor(i3 + 0.5);
    }

    // Compute one pseudo-random hash value for each corner
    float4 hash = permute( permute( permute( 
                float4(i0.z, i1.z, i2.z, i3.z ))
                + float4(i0.y, i1.y, i2.y, i3.y ))
                + float4(i0.x, i1.x, i2.x, i3.x ));

    // Compute generating gradients from a Fibonacci spiral on the unit sphere
    float4 theta = hash * 3.883222077;  // 2*pi/golden ratio
    float4 sz    = hash * -0.006920415 + 0.996539792; // 1-(hash+0.5)*2/289
    float4 psi   = hash * 0.108705628 ; // 10*pi/289, chosen to avoid correlation

    float4 Ct = cos(theta);
    float4 St = sin(theta);
    float4 sz_prime = sqrt( 1.0 - sz*sz ); // s is a point on a unit fib-sphere

    float4 gx, gy, gz;

    // Rotate gradients by angle alpha around a pseudo-random orthogonal axis
#ifdef PSRDNOISE_FAST_ROTATION
    // Fast algorithm, but without dynamic shortcut for alpha = 0
    float4 qx = St;         // q' = norm ( cross(s, n) )  on the equator
    float4 qy = -Ct; 
    float4 qz = float4(0.0, 0.0, 0.0, 0.0);

    float4 px =  sz * qy;   // p' = cross(q, s)
    float4 py = -sz * qx;
    float4 pz = sz_prime;

    psi += alpha;         // psi and alpha in the same plane
    float4 Sa = sin(psi);
    float4 Ca = cos(psi);

    gx = Ca * px + Sa * qx;
    gy = Ca * py + Sa * qy;
    gz = Ca * pz + Sa * qz;
    #else
    // Slightly slower algorithm, but with g = s for alpha = 0, and a
    // strong conditional speedup for alpha = 0 across all fragments
    if(alpha != 0.0) {
        float4 Sp = sin(psi);          // q' from psi on equator
        float4 Cp = cos(psi);

        float4 px = Ct * sz_prime;     // px = sx
        float4 py = St * sz_prime;     // py = sy
        float4 pz = sz;

        float4 Ctp = St*Sp - Ct*Cp;    // q = (rotate( cross(s,n), dot(s,n))(q')
        float4 qx = lerp( Ctp*St, Sp, sz);
        float4 qy = lerp(-Ctp*Ct, Cp, sz);
        float4 qz = -(py*Cp + px*Sp);

        float sin_a = sin(alpha);
        float cos_a = cos(alpha);
        float4 Sa = float4(sin_a, sin_a, sin_a, sin_a);       // psi and alpha in different planes
        float4 Ca = float4(cos_a, cos_a, cos_a, cos_a);

        gx = Ca * px + Sa * qx;
        gy = Ca * py + Sa * qy;
        gz = Ca * pz + Sa * qz;
    }
    else {
        gx = Ct * sz_prime;  // alpha = 0, use s directly as gradient
        gy = St * sz_prime;
        gz = sz;  
    }
#endif

    // Reorganize for dot products below
    float3 g0 = float3(gx.x, gy.x, gz.x);
    float3 g1 = float3(gx.y, gy.y, gz.y);
    float3 g2 = float3(gx.z, gy.z, gz.z);
    float3 g3 = float3(gx.w, gy.w, gz.w);

    // Radial decay with distance from each simplex corner
    float4 w = 0.5 - float4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3));
    w = max(w, 0.0);
    float4 w2 = w * w;
    float4 w3 = w2 * w;

    // The value of the linear ramp from each of the corners
    float4 gdotx = float4(dot(g0,x0), dot(g1,x1), dot(g2,x2), dot(g3,x3));

    // Multiply by the radial decay and sum up the noise value
    float n = dot(w3, gdotx);

    // Compute the first order partial derivatives
    float4 dw = -6.0 * w2 * gdotx;
    float3 dn0 = w3.x * g0 + dw.x * x0;
    float3 dn1 = w3.y * g1 + dw.y * x1;
    float3 dn2 = w3.z * g2 + dw.z * x2;
    float3 dn3 = w3.w * g3 + dw.w * x3;
    gradient = 39.5 * (dn0 + dn1 + dn2 + dn3);

    // Compute the second order partial derivatives
    float4 dw2 = 24.0 * w * gdotx;
    float3 dga0 = dw2.x * x0 * x0 - 6.0 * w2.x * (gdotx.x + 2.0 * g0 * x0);
    float3 dga1 = dw2.y * x1 * x1 - 6.0 * w2.y * (gdotx.y + 2.0 * g1 * x1);
    float3 dga2 = dw2.z * x2 * x2 - 6.0 * w2.z * (gdotx.z + 2.0 * g2 * x2);
    float3 dga3 = dw2.w * x3 * x3 - 6.0 * w2.w * (gdotx.w + 2.0 * g3 * x3);
    dg = 35.0 * (dga0 + dga1 + dga2 + dga3); // (d2n/dx2, d2n/dy2, d2n/dz2)
    float3 dgb0 = dw2.x * x0 * x0.yzx - 6.0 * w2.x * (g0 * x0.yzx + g0.yzx * x0);
    float3 dgb1 = dw2.y * x1 * x1.yzx - 6.0 * w2.y * (g1 * x1.yzx + g1.yzx * x1);
    float3 dgb2 = dw2.z * x2 * x2.yzx - 6.0 * w2.z * (g2 * x2.yzx + g2.yzx * x2);
    float3 dgb3 = dw2.w * x3 * x3.yzx - 6.0 * w2.w * (g3 * x3.yzx + g3.yzx * x3);
    dg2 = 39.5 * (dgb0 + dgb1 + dgb2 + dgb3); // (d2n/dxy, d2n/dyz, d2n/dxz)

    // Scale the return value to fit nicely into the range [-1,1]
    return 39.5 * n;
}

float psrdnoise(float3 x, float3 period, float alpha) {
    float3 g = float3(0.0, 0.0, 0.0);
    return psrdnoise(x, period, alpha, g);
}

float psrdnoise(float3 x, float3 period) {
    return psrdnoise(x, period, 0.0);
}

float psrdnoise(float3 x) {
    return psrdnoise(x, float3(0.0, 0.0, 0.0));
}
#endif