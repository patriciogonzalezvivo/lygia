import glob
import argparse
import os

languages = { 
    'GLSL': {
        'ext': 'glsl',
        'frameworks': ["OpenGL", "WebGL", "Vulkan"],
    },
    'HLSL': {
        'ext': 'hlsl',
        'frameworks': [ "DirectX" ],
    },
    'METAL': {
        'ext': 'msl',
        'frameworks': [ "Metal" ],
    },
    'WGSL': {
        'ext': 'wgsl',
        'frameworks': [ "WebGPU" ],
    },
    'CUDA': {
        'ext': 'cuh',
        'frameworks': [ "CUDA" ],
    },
}

def getAll(extension: str):
    list = []
    for file in glob.glob("**/*." + extension, recursive = True):
        list.append(file);
    list.sort()
    return list


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Prune one or multiple shader languages.')
    argparser.add_argument('--all', '-A', help='Prune all shader languages', action='store_true')
    argparser.add_argument('--keep', '-k', type=str, help='Keep only the specified shader language')
    argparser.add_argument('--remove', '-r', type=str, help='Remove the specified shader language')

    args = argparser.parse_args()

    prune = []
    if args.all:
        prune = list(languages.keys())
    
    if args.keep:
        keep_list = args.keep.upper().split(',')
        print(keep_list)
        for lang in keep_list:
            if lang in prune:
                prune.remove(lang)

    if args.remove:
        prune.append(args.remove)

    print(f"Pruning: {prune}")
    for lang in prune:
        for file in getAll(languages[lang]['ext']):
            os.remove(file)
