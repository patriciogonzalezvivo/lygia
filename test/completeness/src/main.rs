use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

#[derive(Copy, Clone, Debug)]
struct LangInfo
{
    name: &'static str,
    exts: &'static [&'static str],
}

const LANGS: &[LangInfo] = &[
    LangInfo { name: "GLSL", exts: &["glsl", "vert", "frag", "comp"] },
    LangInfo { name: "HLSL", exts: &["hlsl"] },
    LangInfo { name: "Metal", exts: &["msl"] },
    // LangInfo { name: "spirv", exts: &["spv"] },
    LangInfo { name: "WGSL", exts: &["wgsl"] },
    LangInfo { name: "WESL", exts: &["wesl", "wgsl"] },
];

fn main(
    // no args
) {
    let mut all_files = HashSet::<PathBuf>::new();
    let mut files = HashMap::<&str, HashSet<PathBuf>>::new();
    
    for lang in LANGS
    {
        let map = files.entry(lang.name).or_default();

        for ext in lang.exts
        {
            let file_entries = {
                let dirs = match std::env::args().len()
                {
                    0 => vec![".".to_owned()],
                    _ => std::env::args().collect(),
                };

                let mut entries = dirs.into_iter()
                    .flat_map(|arg| glob::glob(&format!("{arg}/**/*.{}", ext)))
                    .flatten()
                    .flatten()
                    .collect::<Vec<_>>();

                entries.sort();
                entries
            };
        
            for entry in file_entries
            {
                let path = entry.as_path();
                let dir = entry.parent().unwrap();
                let stem = path.file_stem().and_then(|s| s.to_str());

                if let Some(name) = stem
                {
                    let file_path = dir.join(name);
                    all_files.insert(file_path.clone());
                    map.insert(file_path);
                }
            }
        }
    }

    // Print table header
    {
        let (header, seperator) = std::iter::once("Module File")
            .chain(LANGS.iter().map(|l| l.name))
            .map(|header| (header.to_owned(), vec!["-"; header.len()].join("")))
            .reduce(|a, b| {
                (format!("{} | {}", a.0, b.0), format!("{} | {}", a.1, b.1))
            })
            .unwrap();

        println!("| {} |", header);
        println!("|:{} |", seperator);
    }

    // Sort module files alphabetically
    let all_files = {
        let mut tmp = all_files.into_iter().collect::<Vec<_>>();
        tmp.sort();
        tmp
    };
    // Print table rows
    for file in all_files
    {
        let items = LANGS.iter()
            .map(|lang| match files.entry(lang.name)
                .or_default()
                .contains(&file)
            {
                true => "OK",
                false => "MISSING",
            })
            .collect::<Vec<_>>()
            .join(" | ");

        println!(
            "| {} | {} |",
            file.to_string_lossy().replace('\\', "/"),
            items
        );
    }
}
