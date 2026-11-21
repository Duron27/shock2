//! Where the System Shock (1994) data lives.

use std::path::PathBuf;

/// Environment variable naming the install folder (the one holding `data/`).
pub const SS1_DATA_PATH: &str = "SS1_DATA_PATH";

/// The game's `data/` folder: `$SS1_DATA_PATH/data` (or `$SS1_DATA_PATH`
/// itself when it already is the data folder), else a `System Shock (1994)*`
/// install beside this repository.
pub fn data_root() -> Option<PathBuf> {
    let has_data = |dir: &PathBuf| dir.join("gamepal.res").is_file();
    if let Some(root) = std::env::var_os(SS1_DATA_PATH).map(PathBuf::from) {
        for dir in [root.join("data"), root.clone()] {
            if has_data(&dir) {
                return Some(dir);
            }
        }
    }
    for parent in ["..", "../..", "../../.."] {
        let Ok(entries) = std::fs::read_dir(parent) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            if name.to_string_lossy().starts_with("System Shock (1994)") {
                let dir = entry.path().join("data");
                if has_data(&dir) {
                    return Some(dir);
                }
            }
        }
    }
    None
}
