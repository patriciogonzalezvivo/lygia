/*
author: Brad Larson
description: adapted version of mean average sampling on four coorners of a sampled point from https://github.com/BradLarson/GPUImage2
use: mean(<sampler2D> texture, <vec2> st, <vec2> pixel)
options:
    MEAN_TYPE: defaults to vec4
    AVERAGE_SAMPLER_FNC(POS_UV): defaults to texture2D(tex,POS_UV)
licence:
    Copyright (c) 2015, Brad Larson. All rights reserved.
    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
    Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    Neither the name of the GPUImage framework nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef MEAN_TYPE
#define MEAN_TYPE vec4
#endif

#ifndef MEAN_AMOUNT
#define MEAN_AMOUNT mean4
#endif

#ifndef MEAN_SAMPLER_FNC
#define MEAN_SAMPLER_FNC(POS_UV) texture2D(tex,POS_UV)
#endif

#ifndef FNC_AVERAGE
#define FNC_AVERAGE
MEAN_TYPE mean4(in sampler2D tex, in vec2 st, in vec2 pixel) {
    MEAN_TYPE topLeft = MEAN_SAMPLER_FNC(st - pixel);
    MEAN_TYPE bottomLeft = MEAN_SAMPLER_FNC(st + vec2(-pixel.x, pixel.y));
    MEAN_TYPE topRight = MEAN_SAMPLER_FNC(st + vec2(pixel.x, -pixel.y));
    MEAN_TYPE bottomRight = MEAN_SAMPLER_FNC(st + pixel);
    return 0.25 * (topLeft + topRight + bottomLeft + bottomRight);
}

MEAN_TYPE mean(in sampler2D tex, in vec2 st, in vec2 pixel) {
    return MEAN_AMOUNT(tex, st, pixel);
}
#endif
