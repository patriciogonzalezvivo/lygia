import os
import sys
import re

def remove_comments(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " " # note: a space and not an empty string
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)

def generate_bundle(root_dir, output_dir):
    header_path = os.path.join(output_dir, 'lygia.h')
    source_path = os.path.join(output_dir, 'lygia.cpp')

    files_map = {}
    
    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Exclude hidden directories/files and build artifacts if present
        if '/.' in dirpath or '\\.' in dirpath:
            continue
            
        for filename in filenames:
            if filename.endswith('.glsl'):
                full_path = os.path.join(dirpath, filename)
                # Calculate relative path from root_dir
                rel_path = os.path.relpath(full_path, root_dir)
                # Ensure forward slashes and prepend 'lygia/'
                key_path = 'lygia/' + rel_path.replace(os.path.sep, '/')
                
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        content = remove_comments(content)
                        files_map[key_path] = content
                except Exception as e:
                    print(f"Skipping {full_path}: {e}")

    # Write Header
    with open(header_path, 'w') as f:
        f.write('#pragma once\n')
        f.write('#include <string>\n\n')
        f.write('// LYGIA, Copyright (c) 2021 Patricio Gonzalez Vivo under Prosperity License - https://prosperitylicense.com/versions/3.0.0\n')
        f.write('// LYGIA, Copyright (c) 2021 Patricio Gonzalez Vivo under Patron License - https://lygia.xyz/license\n')
        f.write('std::string getLygiaFile(const std::string& _path);\n')

    # Write Source
    with open(source_path, 'w') as f:
        f.write('#include "lygia.h"\n')
        f.write('#include <map>\n')
        f.write('#include <vector>\n')
        f.write('#include <cstring>\n')
        f.write('#include <initializer_list>\n\n')
        
        f.write('static std::string _join(const std::initializer_list<const char*>& _parts) {\n')
        f.write('    std::string result;\n')
        f.write('    size_t len = 0;\n')
        f.write('    for (const auto* p : _parts) len += std::strlen(p);\n')
        f.write('    result.reserve(len);\n')
        f.write('    for (const auto* p : _parts) result += p;\n')
        f.write('    return result;\n')
        f.write('}\n\n')

        f.write('std::string getLygiaFile(const std::string& _path) {\n')
        f.write('    static const std::map<std::string, std::string> files = {\n')
        
        for key, content in files_map.items():
            # Use raw string literal with a unique delimiter to avoid conflicts
            # content might contain anything, so we use a delimiter unlikely to be found in GLSL
            delimiter = "LYGIA_CONTENT"
            while delimiter in content:
                delimiter += "_"
            
            # Split content into chunks to avoid C2026 on MSVC (limit around 16k-64k)
            chunk_size = 2048
            chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]

            if len(chunks) == 0:
                f.write(f'        {{"{key}", ""}},\n')
            elif len(chunks) == 1:
                f.write(f'        {{"{key}", R"{delimiter}({chunks[0]}){delimiter}"}},\n')
            else:
                f.write(f'        {{"{key}", _join({{ \n')
                for chunk in chunks:
                    f.write(f'            R"{delimiter}({chunk}){delimiter}",\n')
                f.write(f'        }}) }},\n')
            
        f.write('    };\n')
        f.write('    auto it = files.find(_path);\n')
        f.write('    if (it != files.end()) {\n')
        f.write('        return it->second;\n')
        f.write('    }\n')
        f.write('    return "";\n')
        f.write('}\n')

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python bundle.py <root_dir> <output_dir>")
        sys.exit(1)
    
    generate_bundle(sys.argv[1], sys.argv[2])
