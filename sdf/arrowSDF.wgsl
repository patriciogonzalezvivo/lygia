/*
contributors: TheTurk
description: Distance function for an arrow. Returns float. Originally from https://www.shadertoy.com/view/NttcW8
use: arrowSDF(<vec3> v, <vec3> start, <vec3> end, <float> baseRadius, <float> tipRadius, <float> tipHeight)
*/

fn arrowSDF(v: vec3f, start: vec3f, end: vec3f, baseRadius: f32, tipRadius: f32, tipHeight: f32) -> f32 {
    let t = start - end;
    let l = length(t);
    t /= l;
    l = max(l, tipHeight);
    
    v -= end;
    if(t.y + 1. < .0001) {
        v.y = -v.y;
    } else {
        let k = 1. / (1. + t.y);
        let column1 = vec3f( t.z * t.z * k + t.y, t.x, t.z * -t.x * k );
        let column2 = vec3f( -t.x , t.y , -t.z );
        let column3 = vec3f( -t.x * t.z * k , t.z, t.x * t.x * k + t.y );
        v = mat3x3<f32>( column1 , column2 , column3 ) * v;
    }
    
    let q = vec2f( length(v.xz) , v.y );
    q.x = abs(q.x);
    
    // tip
    let e = vec2f( tipRadius , tipHeight );
    let h = clamp( dot(q,e) / dot(e,e) , 0. , 1. );
    let d1 = q - e * h;
    let d2 = q - vec2f( tipRadius , tipHeight );
    d2.x -= clamp( d2.x , baseRadius - tipRadius , 0. );
    
    // base
    let d3 = q - vec2f( baseRadius , tipHeight );
    d3.y -= clamp( d3.y , 0. , l - tipHeight );
    let d4 = vec2f( q.y - l , max( q.x - baseRadius , 0. ));
    
    float s = max( 
                max( 
                    max( d1.x , -d1.y ), 
                    d4.x
                ), 
                min( d2.y , d3.x ) 
            );

    return sqrt(
             min(
                min( 
                    min(dot(d1,d1), dot(d2,d2)), 
                    dot(d3,d3) 
                ),
                dot(d4,d4)
            )
        ) * sign(s);
}
