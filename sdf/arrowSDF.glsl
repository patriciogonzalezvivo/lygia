/*
contributors: TheTurk
description: Distance function for an arrow. Returns float. Originally from https://www.shadertoy.com/view/NttcW8
use: arrowSDF(<vec3> v, <vec3> start, <vec3> end, <float> baseRadius, <float> tipRadius, <float> tipHeight)
*/

#ifndef FNC_ARROWSDF
#define FNC_ARROWSDF
float arrowSDF(vec3 v, vec3 start, vec3 end, float baseRadius, float tipRadius, float tipHeight){
    vec3 t = start - end;
    float l = length(t);
    t /= l;
    l = max(l, tipHeight);
    
    v -= end;
    if(t.y + 1. < .0001) {
        v.y = -v.y;
    } else {
        float k = 1. / (1. + t.y);
        vec3 column1 = vec3( t.z * t.z * k + t.y, t.x, t.z * -t.x * k );
        vec3 column2 = vec3( -t.x , t.y , -t.z );
        vec3 column3 = vec3( -t.x * t.z * k , t.z, t.x * t.x * k + t.y );
        v = mat3( column1 , column2 , column3 ) * v;
    }
    
    vec2 q = vec2( length(v.xz) , v.y );
    q.x = abs(q.x);
    
    // tip
    vec2 e = vec2( tipRadius , tipHeight );
    float h = clamp( dot(q,e) / dot(e,e) , 0. , 1. );
    vec2 d1 = q - e * h;
    vec2 d2 = q - vec2( tipRadius , tipHeight );
    d2.x -= clamp( d2.x , baseRadius - tipRadius , 0. );
    
    // base
    vec2 d3 = q - vec2( baseRadius , tipHeight );
    d3.y -= clamp( d3.y , 0. , l - tipHeight );
    vec2 d4 = vec2( q.y - l , max( q.x - baseRadius , 0. ));
    
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
#endif
