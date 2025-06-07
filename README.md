<img src="https://lygia.xyz/imgs/lygia.svg" alt="LYGIA" width="200" style="display: block; margin-left: auto; margin-right: auto; filter: drop-shadow(2px 3px 4px gray);">

# LYGIA Shader Library

LYGIA is the biggest shader library. Battle proof, cross-platform and multi-language. Is made of reusable functions that will let you prototype, port and ship projects in just few minutes. It's very granular, flexible and efficient, supports multiple shading languages and can easily be added to virtually any project. There are already integrations for almost all major environments, engines and frameworks.

Best of all, LYGIA grows and improves every day thanks to the support of the community. Become a [Contributor](https://github.com/patriciogonzalezvivo/lygia) or a [![Sponsor](https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&logo=GitHub)](https://github.com/sponsors/patriciogonzalezvivo).

## How to use it?

In your shader just `#include` the functions you need and use them:

<div class="codeAndCanvas" data="example.frag">

    #ifdef GL_ES
    precision mediump float;
    #endif

    uniform vec2    u_resolution;
    uniform float   u_time;

    #include "lygia/space/ratio.glsl"
    #include "lygia/math/decimate.glsl"
    #include "lygia/draw/circle.glsl"

    void main(void) {
        vec3 color = vec3(0.0);
        vec2 st = gl_FragCoord.xy/u_resolution.xy;
        st = ratio(st, u_resolution);
        
        color = vec3(st.x,st.y,abs(sin(u_time)));
        color = decimate(color, 20.);
        color += circle(st, .5, .1);
        
        gl_FragColor = vec4(color, 1.0);
    }
    
</div>

<!-- If you just need to resolve the dependencies of a shader file you got, the fastest way would be to drag&drop your shader file in the box below. We can resolve the dependencies for you.

<div class="container">
    <div class="file-drop-area">
    <span class="file-msg">Drop your shader file <a href="https://lygia.xyz/">here</a></span>
    </div>
</div> -->

LYGIA have been integrated into the following Engines, Frameworks, Creative Tools and online editors:

<p style="text-align: center;" >
    <a href="https://github.com/patriciogonzalezvivo/lygia_unity_examples"><img src="https://lygia.xyz/imgs/unity.png" alt="unity" title="unity" width="64" /></a>
    <a href="https://github.com/franklzt/lygia_unreal_engine_examples"><img src="https://lygia.xyz/imgs/unreal.png" alt="unreal" title="unreal" width="64" /></a>
    <a href="https://www.curseforge.com/minecraft/search?page=1&pageSize=20&sortType=1&search=LYGIA%20Shader%20Library"><img src="https://lygia.xyz/imgs/minecraft.png" alt="minecraft" title="minecraft" width="64" /></a>
    <a href="https://github.com/patriciogonzalezvivo/lygia_examples"><img src="https://lygia.xyz/imgs/glslViewer.png" alt="glslViewer" title="glslViewer" width="64" /></a>
    <a href="https://github.com/irmf/irmf-examples/tree/master/examples/028-lygia"><img src="https://lygia.xyz/imgs/irmf.png" alt="irmf" title="irmf" width="64" /></a>
</p>

<p style="text-align: center;" >
    <a href="https://github.com/guidoschmidt/lygia_threejs_examples"><img src="https://lygia.xyz/imgs/threejs.png" alt="threejs" title="threejs" width="64" /></a>
    <a href="https://github.com/kujohn/lygia_ogl_examples"><img src="https://lygia.xyz/imgs/ogl.png" alt="ogl" title="ogl" width="64" /></a>
    <a href="https://www.npmjs.com/package/lygia"><img src="https://lygia.xyz/imgs/npm.png" alt="npm" title="npm" width="64" /></a>
    <a href="https://codesandbox.io/s/lygia-react-starter-fftx6p"><img src="https://lygia.xyz/imgs/r3f.png" alt="r3rf" title="r3rf" width="64" /></a>
</p>

<p style="text-align: center;" >
    <a href="https://github.com/patriciogonzalezvivo/lygia_p5_examples"><img src="https://lygia.xyz/imgs/p5.png" alt="p5" title="processing" width="64" /></a>
    <a href="https://editor.p5js.org/patriciogonzalezvivo/sketches"><img src="https://lygia.xyz/imgs/p5js.png" alt="p5js" title="p5js" width="64" /></a>
    <a href="https://github.com/patriciogonzalezvivo/lygia_of_examples"><img src="https://lygia.xyz/imgs/of.png" alt="openFrameworks" title="openframeworks" width="64" /></a>
    <a href="https://github.com/vectorsize/lygia-td"><img title="static-resolver by vectorsize" src="https://lygia.xyz/imgs/td.png" alt="touchDesigner" title="touchDesigner" width="64" /></a>
    <a href="https://github.com/patriciogonzalezvivo/comfyui_glslnodes"><img src="https://lygia.xyz/imgs/comfy.png" alt="comfyui" title="comfyUI" width="64" /></a>
    <a href="https://github.com/ossia/score-examples"><img src="https://lygia.xyz/imgs/ossia.png" alt="ossia" title="ossia" width="64" /></a>
</p>

<p style="text-align: center;" >
    <a href="https://www.figma.com/community/plugin/1138854718618193875"><img src="https://lygia.xyz/imgs/figma.png" alt="figma" title="figma" width="64" /></a>
    <a href="https://observablehq.com/@radames/hello-lygia-shader-library"><img src="https://lygia.xyz/imgs/ob.png" alt="observable" title="observable" width="64" /></a>
    <a href="https://www.productioncrate.com/laforge/"><img src="https://lygia.xyz/imgs/laforge.png" alt="laforge" title="laforge" width="64" /></a>
    <a href="https://synesthesia.live/"><img src="https://lygia.xyz/imgs/synesthesia.png" alt="synesthesia" title="synesthesia" width="64" /></a>
    <a href="https://glsl.app/"><img src="https://glsl.app/icon-256.png" alt="glslApp" title="glslApp" width="64"/></a>
    <a href="https://dev.shader.app/"><img src="https://dev.shaders.app/apple-touch-icon.png" alt="shaderApp" title="shaderApp" width="64"/></a>
</p>

If you are working on a project and want to use LYGIA, you have two options: cloning a **local** version that you can bundle into your project; or using the **server** ( https://lygia.xyz ) to resolve the dependencies online.

### LYGIA Locally

If you want to work **locally**, you must ensure that your environment can resolve `#include` dependencies. You can find some examples in [here specifically for GLSL](https://github.com/patriciogonzalezvivo/lygia/blob/main/README_GLSL.md). Then you just need to clone LYGIA into your project relative to the shader you are loading:

```bash
    git clone https://github.com/patriciogonzalezvivo/lygia.git
```

or as a submodule:

```bash
    git submodule add https://github.com/patriciogonzalezvivo/lygia.git
```

Alternatively you may clone LYGIA without the git history and reduce the project size (9MB+) with the following command:

```bash
    npx degit https://github.com/patriciogonzalezvivo/lygia.git lygia
```

If you are concerned about the size of the library you might also be interested on pruning the library to only the language you are using. You can do that by using the `prune.py` script. For example:

```bash
    python prune.py --all --keep glsl
```

Alternatively, if you are working on a `npm` project, there is a [npm bundle](https://www.npmjs.com/package/lygia) you could use.

If you are working on a web project you may want to resolve the dependencies using a bundler like [vite glsl plugin (local bundle)](https://github.com/UstymUkhman/vite-plugin-glsl), [esbuild glsl plugin (local bundle)](https://github.com/ricardomatias/esbuild-plugin-glsl-include) or [webpack glsl plugin (local bundle)](https://github.com/grieve/webpack-glsl-loader).

### LYGIA server

If you are working on a **cloud platform** (like [CodePen](https://codepen.io/) or [Observable](https://observablehq.com/@radames/hello-lygia-shader-library)) you probably want to resolve the dependencies without needing to install anything. For that just add a link to `https://lygia.xyz/resolve.js` (JS) or `https://lygia.xyz/resolve.esm.js` (ES6 module):

```html
    <!-- As JavaScript source -->
    <script src="https://lygia.xyz/resolve.js"></script>

    <!-- Or as an ES6 module -->
    <script type="module">
        import resolveLygia from "https://lygia.xyz/resolve.esm.js"
    </script>
```

Then you can resolve the dependencies by passing a `string` or `strings[]` to `resolveLygia()` or `resolveLygiaAsync()`:

```js
    // 1. FIRST

    // Sync resolver, one include at a time
    vertSource = resolveLygia(vertSource);
    fragSource = resolveLygia(fragSource);

    // OR.
    
    // Async resolver, all includes in parallel calls
    vertSource = resolveLygiaAsync(vertSource);
    fragSource = resolveLygiaAsync(fragSource);
    
    // 2. SECOND

    // Use the resolved source code 
    shdr = createShader(vertSource, fragSource);
```

This function can also resolve dependencies to previous versions of LYGIA by using this pattern `lygia/vX.X/...` or `lygia/vX.X.X/...` on you dependency paths. For example:

```glsl
#include "lygia/v1.0/math/decimation.glsl"
#include "lygia/v1.2.1/math/decimation.glsl"
```

### How is LYGIA organized?

The functions are divided into different categories:

* [`math/`](https://lygia.xyz/math): general math functions and constants: `PI`, `SqrtLength()`, etc. 
* [`space/`](https://lygia.xyz/space): general spatial operations: `scale()`, `rotate()`, etc. 
* [`color/`](https://lygia.xyz/color): general color operations: `luma()`, `saturation()`, blend modes, palettes, color space conversion, and tonemaps.
* [`animation/`](https://lygia.xyz/animation): animation operations: easing
* [`generative/`](https://lygia.xyz/generative): generative functions: `random()`, `noise()`, etc. 
* [`sdf/`](https://lygia.xyz/sdf): signed distance field functions.
* [`draw/`](https://lygia.xyz/draw): drawing functions like `digits()`, `stroke()`, `fill`, etc.
* [`sample/`](https://lygia.xyz/sample): sample operations
* [`filter/`](https://lygia.xyz/filter): typical filter operations: different kind of blurs, mean and median filters.
* [`distort/`](https://lygia.xyz/distort): distort sampling operations
* [`lighting/`](https://lygia.xyz/lighting): different lighting models and functions for forward/deferred/raymarching rendering
* [`geometry/`](https://lygia.xyz/geometry): operation related to geometries: intersections and AABB accelerating structures.
* [`morphological/`](https://lygia.xyz/morphological): morphological filters: dilation, erosion, alpha and poisson fill.

### How is it [designed](https://github.com/patriciogonzalezvivo/lygia/blob/main/DESIGN.md)?

LYGIA is designed to be very granular (each file holds one function), multi-language (each language has its own file extension) and flexible. Flexible how?
There are some functions whose behavior can be changed using the `#define` keyword before including them. For example, [gaussian blurs](filter/gaussianBlur) are usually done in two passes. By default, these are performed on their 1D version, but if you are interested in using a 2D kernel, all in the same pass, you will need to add the `GAUSSIANBLUR_2D` keyword, as follows:

```glsl

    #define GAUSSIANBLUR_2D
    #include "filter/gaussianBlur.glsl"

    void main(void) {
        ...
        
        vec2 pixel = 1./u_resolution;
        color = gaussianBlur(u_tex0, uv, pixel, 9);
        ...
    }

```

In the same way you can change the sampling function that the gaussian uses. Ex:

```glsl
// from
#define GAUSSIANBLUR_SAMPLER_FNC(TEX, UV) texture2D(TEX, UV)
// to 
#include "lygia/sample/clamp2edges.glsl"
#define GAUSSIANBLUR_SAMPLER_FNC(TEX, UV) sampleClamp2edge(TEX, UV)
```

Learn more about [LYGIA's design principles in the DESIGN.md file](https://github.com/patriciogonzalezvivo/lygia/blob/main/DESIGN.md).

## Contributions

LYGIA has a long way to go and welcomes all kinds of contributions. You can help by:

* **Bug fixing**
* **Translation**, keeping parity between languages (GLSL, HLSL, MSL, WGSL, TSL, CUDA, OSL, etc.) is a big part of the challenge. Not all languages are the same and we want to make sure each function is optimized and carefully crafted for each environment. This means, the more eyes are looking at this, the better. Please make sure to read and understand the [Design Principles](https://github.com/patriciogonzalezvivo/lygia/blob/main/DESIGN.md) before starting.
* **New functions or improving the current implementations**. Please take a look to the [Contribution Guidelines](https://github.com/patriciogonzalezvivo/lygia/blob/main/CONTRIBUTE.md) before starting.
* **Documentation**. Each function has a header with some information describing the function. Make sure to fill this information when adding a new function.
* Adding new **examples** and integrations for new environments like: [Godot](https://godotengine.org/), [ISF](https://isf.video/), [MaxMSP](https://cycling74.com/products/max), etc.
* **Financial** [sponsorships](https://github.com/sponsors/patriciogonzalezvivo). Right now, the money that flows in is invested on the server and infrastructure. Long term plan will be to be able to pay lead contributors and maintainers.

Collaborators and sponsors are automatically added to the [commercial license](https://lygia.xyz/license). Making a PR or subscribing to the GitHub sponsors program is the shortest path to get access to the commercial license. It's all automated, not red taping. LYGIA belongs to those that take care of it.

## License

LYGIA belongs to those that support it. For that it is dual-licensed under the [Prosperity License](https://prosperitylicense.com/versions/3.0.0) and the [Patron License](https://lygia.xyz/license) for [sponsors](https://github.com/sponsors/patriciogonzalezvivo) and [contributors](https://github.com/patriciogonzalezvivo/lygia/graphs/contributors).

[Sponsors](https://github.com/sponsors/patriciogonzalezvivo) and [contributors](https://github.com/patriciogonzalezvivo/lygia/graphs/contributors) are automatically added to the [Patron License](https://lygia.xyz/license) and they can ignore any non-commercial rule of the [Prosperity License](https://prosperitylicense.com/versions/3.0.0) software.

It's also possible to get a permanent commercial license hooked to a single and specific version of LYGIA.

If you have doubts please reach out to patriciogonzalezvivo at gmail dot com

## Credits

Created and maintained by [Patricio Gonzalez Vivo](https://patriciogonzalezvivo.com/) ( <a rel="me" href="https://merveilles.town/@patricio">Mastodon</a> | [Twitter](https://twitter.com/patriciogv) | [Instagram](https://www.instagram.com/patriciogonzalezvivo/) | [GitHub](https://github.com/sponsors/patriciogonzalezvivo) ) and every direct or indirect [contributors](https://github.com/patriciogonzalezvivo/lygia/graphs/contributors) to the GitHub repository.

This library has been built in many cases on top of the work of brilliant and generous people like: [Inigo Quiles](https://www.iquilezles.org/), [Morgan McGuire](https://casual-effects.com/), [Alan Wolfe](https://blog.demofox.org/), [Matt DesLauriers](https://www.mattdesl.com/), [Bjorn Ottosson](https://github.com/bottosson), [Hugh Kennedy](https://github.com/hughsk), and many others.

It also is being constantly maintained, translated and/or extended by generous contributors like: [Shadi El Hajj](https://github.com/shadielhajj), [Kathy](https://github.com/kfahn22), [Bonsak Schiledrop](https://github.com/bonsak), [Amin Shazrin](https://github.com/ammein), [Guido Schmidt](https://github.com/guidoschmidt), and many others.
