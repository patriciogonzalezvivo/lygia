/*
contributors:  Inigo Quiles
description: generate the SDF of a cone
use: <float> coneSDF( in <vec3> pos, in <vec3> c ) 
*/

fn coneSDF3(p: vec3f, c: vec3f) -> f32 {
    let q = vec2f( length(p.xz), p.y );
    let d1 = -q.y-c.z;
    let d2 = max( dot(q,c.xy), q.y);
    return length(max(vec2f(d1,d2),0.0)) + min(max(d1,d2), 0.);
}

// vertical
fn coneSDF3a(p: vec3f, c: vec2f, h: f32) -> f32 {
    let q = h*vec2f(c.x,-c.y)/c.y;
    let w = vec2f( length(p.xz), p.y );
    
	let a = w - q*clamp( dot(w,q)/dot(q,q), 0.0, 1.0 );
    let b = w - q*vec2f( clamp( w.x/q.x, 0.0, 1.0 ), 1.0 );
    let k = sign( q.y );
    let d = min(dot( a, a ),dot(b, b));
    let s = max( k*(w.x*q.y-w.y*q.x),k*(w.y-q.y)  );
	return sqrt(d)*sign(s);
}

// Round
fn coneSDF3b(p: vec3f, r1: f32, r2: f32, h: f32) -> f32 {
    let q = vec2f( length(p.xz), p.y );
    
    let b = (r1-r2)/h;
    let a = sqrt(1.0-b*b);
    let k = dot(q,vec2f(-b,a));
    
    if( k < 0.0 ) return length(q) - r1;
    if( k > a*h ) return length(q-vec2f(0.0,h)) - r2;
        
    return dot(q, vec2f(a,b) ) - r1;
}
