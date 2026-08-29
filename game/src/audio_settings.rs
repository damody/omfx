//! Frontend audio settings persisted beside the executor binary.

use std::path::{Path, PathBuf};

use toml_edit::{table, value, DocumentMut};

pub const DEFAULT_MUSIC_VOLUME: f32 = 0.0;
const CONFIG_FILE_NAME: &str = "config.toml";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AudioSettings {
    pub music_volume: f32,
}

impl Default for AudioSettings {
    fn default() -> Self {
        Self {
            music_volume: DEFAULT_MUSIC_VOLUME,
        }
    }
}

impl AudioSettings {
    pub fn load_or_create() -> Self {
        let Ok(path) = config_path() else {
            log::warn!("無法取得執行檔路徑，音樂音量使用預設值 0%");
            return Self::default().muted_for_startup();
        };

        let settings = match load_or_create_at(&path) {
            Ok(settings) => settings,
            Err(err) => {
                log::warn!("讀取音效設定失敗（{}）：{}", path.display(), err);
                Self::default()
            }
        };

        settings.muted_for_startup()
    }

    pub fn save(self) {
        let Ok(path) = config_path() else {
            log::warn!("無法取得執行檔路徑，未儲存音樂音量");
            return;
        };

        if let Err(err) = save_at(&path, self) {
            log::warn!("儲存音效設定失敗（{}）：{}", path.display(), err);
        }
    }

    fn muted_for_startup(mut self) -> Self {
        self.music_volume = 0.0;
        self
    }
}

fn config_path() -> std::io::Result<PathBuf> {
    let executable = std::env::current_exe()?;
    let directory = executable
        .parent()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "執行檔沒有父目錄"))?;
    Ok(directory.join(CONFIG_FILE_NAME))
}

fn load_or_create_at(path: &Path) -> Result<AudioSettings, String> {
    if !path.exists() {
        let settings = AudioSettings::default();
        save_at(path, settings)?;
        log::info!("已建立音效設定檔：{}", path.display());
        return Ok(settings);
    }

    let text = std::fs::read_to_string(path).map_err(|err| err.to_string())?;
    let document = text.parse::<DocumentMut>().map_err(|err| err.to_string())?;
    let raw = document["audio"]["music_volume"]
        .as_float()
        .or_else(|| {
            document["audio"]["music_volume"]
                .as_integer()
                .map(|value| value as f64)
        })
        .ok_or_else(|| "缺少 [audio].music_volume 數值".to_string())?;

    if !raw.is_finite() {
        return Err("[audio].music_volume 必須是有限數值".to_string());
    }

    Ok(AudioSettings {
        music_volume: (raw as f32).clamp(0.0, 1.0),
    })
}

fn save_at(path: &Path, settings: AudioSettings) -> Result<(), String> {
    let mut document = if path.exists() {
        std::fs::read_to_string(path)
            .map_err(|err| err.to_string())?
            .parse::<DocumentMut>()
            .map_err(|err| err.to_string())?
    } else {
        DocumentMut::new()
    };

    if !document.as_table().contains_key("audio") {
        document["audio"] = table();
    }
    let stored_volume = settings
        .music_volume
        .clamp(0.0, 1.0)
        .to_string()
        .parse::<f64>()
        .map_err(|err| err.to_string())?;
    document["audio"]["music_volume"] = value(stored_volume);
    std::fs::write(path, document.to_string()).map_err(|err| err.to_string())?;
    log::info!("音效設定已儲存至：{}", path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "omfx_audio_settings_{}_{}_config.toml",
            std::process::id(),
            name
        ))
    }

    #[test]
    fn missing_config_is_created_muted() {
        let path = test_path("create");
        let _ = std::fs::remove_file(&path);

        let settings = load_or_create_at(&path).expect("settings are created");
        let text = std::fs::read_to_string(&path).expect("config is readable");

        assert_eq!(settings.music_volume, DEFAULT_MUSIC_VOLUME);
        assert!(text.contains("[audio]"));
        assert!(text.contains("music_volume = 0.0\n"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn stored_volume_is_muted_for_startup_without_rewriting_config() {
        let path = test_path("startup_muted");
        std::fs::write(
            &path,
            "[audio]\nmusic_volume = 0.65\n[video]\nfullscreen = true\n",
        )
        .expect("seed config");

        let settings = load_or_create_at(&path)
            .expect("settings load")
            .muted_for_startup();
        let text = std::fs::read_to_string(&path).expect("config is readable");

        assert_eq!(settings.music_volume, 0.0);
        assert!(text.contains("music_volume = 0.65"));
        assert!(text.contains("fullscreen = true"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn saved_volume_is_loaded_and_other_settings_are_preserved() {
        let path = test_path("round_trip");
        std::fs::write(&path, "[video]\nfullscreen = true\n").expect("seed config");

        save_at(&path, AudioSettings { music_volume: 0.65 }).expect("settings save");
        let settings = load_or_create_at(&path).expect("settings load");
        let text = std::fs::read_to_string(&path).expect("config is readable");

        assert!((settings.music_volume - 0.65).abs() < f32::EPSILON);
        assert!(text.contains("fullscreen = true"));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn loaded_volume_is_clamped_to_valid_range() {
        let path = test_path("clamp");
        std::fs::write(&path, "[audio]\nmusic_volume = 5.0\n").expect("seed config");

        let settings = load_or_create_at(&path).expect("settings load");

        assert_eq!(settings.music_volume, 1.0);
        let _ = std::fs::remove_file(path);
    }
}
