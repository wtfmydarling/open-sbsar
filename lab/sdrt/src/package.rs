//! Reading .sbsar archives: the 7z container (in process, no external tool), its graphs (one
//! compiled program `.sbsasm` and one description `.xml` per graph) and the XML metadata.
use std::io::Cursor;

/// One graph of an archive: the compiled program and its XML description.
pub struct RawGraph {
    /// Path of the program inside the archive, e.g. `assemblies/content/0000/001.sbsasm`.
    pub path: String,
    pub asm: Vec<u8>,
    pub xml: String,
}

/// Every graph of a .sbsar (a 7z archive), in archive order. Other members are ignored.
pub fn read_archive(bytes: &[u8]) -> Result<Vec<RawGraph>, String> {
    let mut reader = sevenz_rust2::ArchiveReader::new(Cursor::new(bytes), sevenz_rust2::Password::empty())
        .map_err(|e| format!("not a readable .sbsar (7z) archive: {e}"))?;
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();
    reader
        .for_each_entries(|entry, r| {
            let name = entry.name().replace('\\', "/");
            if !entry.is_directory() && (name.ends_with(".sbsasm") || name.ends_with(".xml")) {
                let mut data = Vec::new();
                r.read_to_end(&mut data)?;
                files.push((name, data));
            } else {
                std::io::copy(r, &mut std::io::sink())?;
            }
            Ok(true)
        })
        .map_err(|e| format!("cannot extract the .sbsar archive: {e}"))?;
    let mut graphs = Vec::new();
    for (name, data) in &files {
        let Some(stem) = name.strip_suffix(".sbsasm") else { continue };
        // the description next to the program: same stem, else the only XML of the same folder
        let dir = stem.rsplit_once('/').map_or("", |(d, _)| d);
        let xml = files
            .iter()
            .find(|(n, _)| n.strip_suffix(".xml") == Some(stem))
            .or_else(|| {
                let same: Vec<_> = files.iter().filter(|(n, _)| n.ends_with(".xml") && n.rsplit_once('/').map_or("", |(d, _)| d) == dir).collect();
                (same.len() == 1).then(|| same[0])
            })
            .map(|(_, x)| String::from_utf8_lossy(x).into_owned())
            .unwrap_or_default();
        graphs.push(RawGraph { path: name.clone(), asm: data.clone(), xml });
    }
    if graphs.is_empty() {
        return Err("the archive holds no compiled graph (.sbsasm)".into());
    }
    Ok(graphs)
}

/// The program and its description from a file: a .sbsar (its first graph), or an .sbsasm next
/// to an .xml of the same name.
pub fn load_path(path: &std::path::Path) -> Result<(Vec<u8>, String), String> {
    if path.extension().is_some_and(|e| e == "sbsasm") {
        let b = std::fs::read(path).map_err(|e| e.to_string())?;
        let xml = std::fs::read_to_string(path.with_extension("xml")).unwrap_or_default();
        return Ok((b, xml));
    }
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let g = read_archive(&bytes)?.into_iter().next().unwrap();
    Ok((g.asm, g.xml))
}

/// uid of each input by identifier in the XML.
pub fn input_uids(xml: &str) -> Vec<(String, u32)> {
    let mut v = vec![];
    for part in xml.split("<input ").skip(1) {
        let attr = |name: &str| part.split(&format!("{name}=\"")).nth(1).and_then(|s| s.split('"').next()).map(str::to_string);
        if let (Some(uid), Some(id)) = (attr("uid"), attr("identifier")) {
            if let Ok(u) = uid.parse() {
                v.push((id, u));
            }
        }
    }
    v
}

/// String inputs (XML type 6): their defaults, XML entities decoded.
pub fn string_defaults(xml: &str) -> Vec<(u32, String)> {
    let mut v = vec![];
    for part in xml.split("<input ").skip(1) {
        let head = format!(" {}", part.split('>').next().unwrap_or("")).replace(['\n', '\t', '\r'], " ");
        let attr = |name: &str| head.split(&format!(" {name}=\"")).nth(1).and_then(|s| s.split('"').next()).map(str::to_string);
        if attr("type").as_deref() != Some("6") {
            continue;
        }
        if let (Some(uid), Some(d)) = (attr("uid").and_then(|u| u.parse::<u32>().ok()), attr("default")) {
            v.push((uid, xml_unescape(&d)));
        }
    }
    v
}

/// XML attribute text: the five named entities and numeric character references.
pub fn xml_unescape(s: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        let tail = &rest[i..];
        let Some(j) = tail.find(';') else {
            out.push_str(tail);
            return out;
        };
        let ent = &tail[1..j];
        let c = match ent {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            _ if ent.starts_with("#x") || ent.starts_with("#X") => u32::from_str_radix(&ent[2..], 16).ok().and_then(char::from_u32),
            _ if ent.starts_with('#') => ent[1..].parse().ok().and_then(char::from_u32),
            _ => None,
        };
        match c {
            Some(c) => out.push(c),
            None => out.push_str(&tail[..=j]),
        }
        rest = &tail[j + 1..];
    }
    out.push_str(rest);
    out
}

/// uid → identifier of each output in the XML.
pub fn output_names(xml: &str) -> Vec<(u32, String)> {
    let mut v = vec![];
    for part in xml.split("<output ").skip(1) {
        let attr = |name: &str| part.split(&format!("{name}=\"")).nth(1).and_then(|s| s.split('"').next()).map(str::to_string);
        if let (Some(uid), Some(id)) = (attr("uid"), attr("identifier")) {
            if let Ok(u) = uid.parse() {
                v.push((u, id));
            }
        }
    }
    v
}

/// The kind of a graph input (XML `type`).
#[derive(Clone, Debug, PartialEq)]
pub enum InputKind {
    /// 1 to 4 floats (types 0..3)
    Float(usize),
    /// 1 to 4 integers (types 4, 8, 9, 10); booleans are 1 integer with a toggle widget
    Int(usize),
    /// an image input (type 5)
    Image,
    /// a string (type 6)
    String,
    /// a type this library does not know
    Other(u32),
}

/// A graph input as the XML describes it.
#[derive(Clone, Debug)]
pub struct InputInfo {
    pub identifier: String,
    pub uid: u32,
    pub kind: InputKind,
    /// the XML default as written (e.g. "0.5" or "8,8"); the program's own default is applied
    /// when the input is not set
    pub default: Option<String>,
    pub label: Option<String>,
    pub group: Option<String>,
    /// the GUI widget: slider, color, togglebutton, combobox, angle, position, image, ...
    pub widget: Option<String>,
    pub min: Option<Vec<f64>>,
    pub max: Option<Vec<f64>>,
    /// combobox entries: (value, text)
    pub options: Vec<(i32, String)>,
}

/// A graph output as the XML describes it.
#[derive(Clone, Debug)]
pub struct OutputInfo {
    pub identifier: String,
    pub uid: u32,
    pub label: Option<String>,
    pub group: Option<String>,
    /// usages such as baseColor, normal, roughness (from `<usage name=...>` and `channel names`)
    pub usages: Vec<String>,
    pub colorspace: Option<String>,
    /// an image output (XML type 5); other outputs are values
    pub is_image: bool,
}

fn nums(s: &str) -> Option<Vec<f64>> {
    let v: Vec<f64> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
    (!v.is_empty()).then_some(v)
}

/// The inputs and outputs of a graph from its XML description (empty lists if there is none).
pub fn metadata(xml: &str) -> (Option<String>, Vec<InputInfo>, Vec<OutputInfo>) {
    let opts = roxmltree::ParsingOptions { allow_dtd: true, ..Default::default() };
    let Ok(doc) = roxmltree::Document::parse_with_options(xml, opts) else { return (None, vec![], vec![]) };
    let label = doc.descendants().find(|n| n.has_tag_name("graph")).and_then(|g| g.attribute("label").map(str::to_string));
    let mut inputs = vec![];
    for n in doc.descendants().filter(|n| n.has_tag_name("input")) {
        let (Some(id), Some(uid)) = (n.attribute("identifier"), n.attribute("uid").and_then(|u| u.parse().ok())) else { continue };
        let ty: u32 = n.attribute("type").and_then(|t| t.parse().ok()).unwrap_or(u32::MAX);
        let kind = match ty {
            0..=3 => InputKind::Float(ty as usize + 1),
            4 => InputKind::Int(1),
            8..=10 => InputKind::Int(ty as usize - 6),
            5 => InputKind::Image,
            6 => InputKind::String,
            t => InputKind::Other(t),
        };
        let gui = n.children().find(|c| c.has_tag_name("inputgui"));
        let range = gui.and_then(|g| g.descendants().find(|c| c.attribute("min").is_some() || c.attribute("max").is_some()));
        let options = gui
            .map(|g| {
                g.descendants()
                    .filter(|c| c.has_tag_name("guicomboboxitem"))
                    .filter_map(|c| Some((c.attribute("value")?.parse().ok()?, c.attribute("text").unwrap_or("").to_string())))
                    .collect()
            })
            .unwrap_or_default();
        inputs.push(InputInfo {
            identifier: id.to_string(),
            uid,
            kind,
            default: n.attribute("default").map(str::to_string),
            label: gui.and_then(|g| g.attribute("label")).map(str::to_string),
            group: gui.and_then(|g| g.attribute("group")).map(str::to_string),
            widget: gui.and_then(|g| g.attribute("widget")).map(str::to_string),
            min: range.and_then(|r| r.attribute("min")).and_then(nums),
            max: range.and_then(|r| r.attribute("max")).and_then(nums),
            options,
        });
    }
    let mut outputs = vec![];
    for n in doc.descendants().filter(|n| n.has_tag_name("output")) {
        let (Some(id), Some(uid)) = (n.attribute("identifier"), n.attribute("uid").and_then(|u| u.parse().ok())) else { continue };
        let gui = n.children().find(|c| c.has_tag_name("outputgui"));
        let mut usages: Vec<String> = n.descendants().filter(|c| c.has_tag_name("usage")).filter_map(|c| c.attribute("name").map(str::to_string)).collect();
        let channels: Vec<_> = n.descendants().filter(|c| c.has_tag_name("channel")).collect();
        for c in &channels {
            if let Some(names) = c.attribute("names") {
                usages.extend(names.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()));
            }
        }
        usages.dedup();
        outputs.push(OutputInfo {
            identifier: id.to_string(),
            uid,
            label: gui.and_then(|g| g.attribute("label")).map(str::to_string),
            group: gui.and_then(|g| g.attribute("group")).map(str::to_string),
            usages,
            colorspace: channels.iter().find_map(|c| c.attribute("colorspace")).map(str::to_string),
            is_image: n.attribute("type").is_none_or(|t| t == "5"),
        });
    }
    (label, inputs, outputs)
}
