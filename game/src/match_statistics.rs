use serde_json::{Map, Value};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatchResult {
    Victory,
    Defeat,
    Abandoned,
}

impl MatchResult {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Victory => "victory",
            Self::Defeat => "defeat",
            Self::Abandoned => "abandoned",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatchSettlement {
    pub result: MatchResult,
    pub highest_wave: u32,
    pub match_kills: u32,
}

#[derive(Clone, Debug, Default)]
pub struct MatchSessionTracker {
    started: bool,
    settled: bool,
    highest_wave: u32,
    match_kills: u32,
    result: Option<MatchResult>,
}

impl MatchSessionTracker {
    pub fn start(&mut self) {
        *self = Self {
            started: true,
            ..Self::default()
        };
    }

    pub fn observe(&mut self, round: u32, match_kills: u32) {
        if !self.started || self.settled {
            return;
        }
        self.highest_wave = self.highest_wave.max(round);
        self.match_kills = self.match_kills.max(match_kills);
    }

    pub fn mark_terminal(&mut self, result: MatchResult) {
        if self.started && !self.settled && self.result.is_none() {
            self.result = Some(result);
        }
    }

    pub fn take_settlement(&mut self) -> Option<MatchSettlement> {
        if !self.started || self.settled {
            return None;
        }
        self.settled = true;
        Some(MatchSettlement {
            result: self.result.unwrap_or(MatchResult::Abandoned),
            highest_wave: self.highest_wave,
            match_kills: self.match_kills,
        })
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProfileStatistics {
    pub games_played: u32,
    pub wins: u32,
    pub highest_wave: u32,
    pub total_kills: u32,
}

pub fn default_profile_path() -> PathBuf {
    PathBuf::from("omb").join("player_profile.json")
}

pub fn player_visible_wave(round: u32, total_rounds: u32, round_is_running: bool) -> u32 {
    if round_is_running {
        round.saturating_add(1).min(total_rounds.max(1))
    } else {
        round
    }
}

pub fn settle_profile(
    path: &Path,
    settlement: MatchSettlement,
) -> Result<ProfileStatistics, String> {
    settle_profile_with_replacer(path, settlement, replace_file)
}

fn settle_profile_with_replacer<F>(
    path: &Path,
    settlement: MatchSettlement,
    replacer: F,
) -> Result<ProfileStatistics, String>
where
    F: FnOnce(&Path, &Path) -> std::io::Result<()>,
{
    let mut profile = read_profile_value(path)?;
    let stats = merge_settlement(&mut profile, settlement)?;
    let bytes = serde_json::to_vec_pretty(&profile).map_err(|error| {
        format!(
            "serialize match statistics result={} path={}: {error}",
            settlement.result.as_str(),
            path.display()
        )
    })?;
    let temporary = temporary_path(path);
    let write_result = (|| -> std::io::Result<()> {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        drop(file);
        replacer(&temporary, path)
    })();
    if let Err(error) = write_result {
        let _ = fs::remove_file(&temporary);
        return Err(format!(
            "persist match statistics result={} path={}: {error}",
            settlement.result.as_str(),
            path.display()
        ));
    }
    Ok(stats)
}

fn read_profile_value(path: &Path) -> Result<Value, String> {
    match fs::read(path) {
        Ok(bytes) => serde_json::from_slice(&bytes)
            .map_err(|error| format!("parse profile {}: {error}", path.display())),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Ok(Value::Object(default_profile_object()))
        }
        Err(error) => Err(format!("read profile {}: {error}", path.display())),
    }
}

fn default_profile_object() -> Map<String, Value> {
    Map::from_iter([
        ("total_kp".to_string(), Value::from(20u32)),
        ("spent_kp".to_string(), Value::from(0u32)),
        ("unlocked_nodes".to_string(), Value::Array(Vec::new())),
        ("enabled".to_string(), Value::Bool(true)),
    ])
}

fn merge_settlement(
    profile: &mut Value,
    settlement: MatchSettlement,
) -> Result<ProfileStatistics, String> {
    let object = profile
        .as_object_mut()
        .ok_or_else(|| "player profile root must be a JSON object".to_string())?;
    let games_played = profile_u32(object, "games_played").saturating_add(1);
    let wins = profile_u32(object, "wins")
        .saturating_add(u32::from(matches!(settlement.result, MatchResult::Victory)));
    let highest_wave = profile_u32(object, "highest_wave").max(settlement.highest_wave);
    let total_kills = profile_u32(object, "total_kills").saturating_add(settlement.match_kills);
    for (key, value) in [
        ("games_played", games_played),
        ("wins", wins),
        ("highest_wave", highest_wave),
        ("total_kills", total_kills),
    ] {
        object.insert(key.to_string(), Value::from(value));
    }
    Ok(ProfileStatistics {
        games_played,
        wins,
        highest_wave,
        total_kills,
    })
}

fn profile_u32(object: &Map<String, Value>, key: &str) -> u32 {
    object
        .get(key)
        .and_then(Value::as_u64)
        .map(|value| value.min(u64::from(u32::MAX)) as u32)
        .unwrap_or(0)
}

fn temporary_path(path: &Path) -> PathBuf {
    static SERIAL: AtomicU64 = AtomicU64::new(0);
    let serial = SERIAL.fetch_add(1, Ordering::Relaxed);
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("player_profile.json");
    path.with_file_name(format!(
        ".{file_name}.match-stats.{}.{}.tmp",
        std::process::id(),
        serial
    ))
}

#[cfg(not(windows))]
fn replace_file(temporary: &Path, destination: &Path) -> std::io::Result<()> {
    fs::rename(temporary, destination)
}

#[cfg(windows)]
fn replace_file(temporary: &Path, destination: &Path) -> std::io::Result<()> {
    if !destination.exists() {
        return fs::rename(temporary, destination);
    }
    use std::os::windows::ffi::OsStrExt;
    use std::ptr;

    #[link(name = "Kernel32")]
    unsafe extern "system" {
        fn ReplaceFileW(
            replaced_file_name: *const u16,
            replacement_file_name: *const u16,
            backup_file_name: *const u16,
            replace_flags: u32,
            exclude: *mut std::ffi::c_void,
            reserved: *mut std::ffi::c_void,
        ) -> i32;
    }

    const REPLACEFILE_WRITE_THROUGH: u32 = 0x0000_0001;
    let destination_wide: Vec<u16> = destination
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let temporary_wide: Vec<u16> = temporary
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let success = unsafe {
        ReplaceFileW(
            destination_wide.as_ptr(),
            temporary_wide.as_ptr(),
            ptr::null(),
            REPLACEFILE_WRITE_THROUGH,
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    if success == 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "omfx-match-statistics-{name}-{}-{}",
            std::process::id(),
            temporary_path(Path::new("serial")).display()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn settlement(result: MatchResult) -> MatchSettlement {
        MatchSettlement {
            result,
            highest_wave: 28,
            match_kills: 321,
        }
    }

    #[test]
    fn tracker_settles_victory_once_and_defaults_exit_to_abandoned() {
        let mut tracker = MatchSessionTracker::default();
        assert_eq!(
            tracker.take_settlement(),
            None,
            "startup failure is not played"
        );
        tracker.start();
        tracker.observe(12, 40);
        tracker.observe(28, 321);
        tracker.mark_terminal(MatchResult::Victory);
        tracker.mark_terminal(MatchResult::Defeat);
        assert_eq!(
            tracker.take_settlement(),
            Some(settlement(MatchResult::Victory))
        );
        assert_eq!(tracker.take_settlement(), None);

        tracker.start();
        tracker.observe(17, 9);
        assert_eq!(
            tracker.take_settlement(),
            Some(MatchSettlement {
                result: MatchResult::Abandoned,
                highest_wave: 17,
                match_kills: 9,
            })
        );
    }

    #[test]
    fn running_round_records_the_player_visible_wave() {
        assert_eq!(player_visible_wave(0, 40, true), 1);
        assert_eq!(player_visible_wave(16, 40, true), 17);
        assert_eq!(player_visible_wave(40, 40, false), 40);
        assert_eq!(player_visible_wave(u32::MAX, 40, true), 40);
    }

    #[test]
    fn victory_defeat_and_abandon_merge_without_losing_profile_fields() {
        let dir = test_dir("merge");
        let path = dir.join("player_profile.json");
        fs::write(
            &path,
            r#"{"total_kp":25,"spent_kp":2,"unlocked_nodes":["x"],"unknown":"keep","highest_wave":40}"#,
        )
        .unwrap();

        let victory = settle_profile(&path, settlement(MatchResult::Victory)).unwrap();
        assert_eq!(victory.games_played, 1);
        assert_eq!(victory.wins, 1);
        assert_eq!(victory.highest_wave, 40);
        assert_eq!(victory.total_kills, 321);
        let defeat = settle_profile(&path, settlement(MatchResult::Defeat)).unwrap();
        let abandoned = settle_profile(&path, settlement(MatchResult::Abandoned)).unwrap();
        assert_eq!(abandoned.games_played, 3);
        assert_eq!(abandoned.wins, 1);
        assert_eq!(defeat.wins, 1);
        let saved: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(saved["total_kp"], 25);
        assert_eq!(saved["unknown"], "keep");
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn legacy_profile_and_counters_use_defaults_and_saturation() {
        let mut profile = serde_json::json!({
            "total_kp": 20,
            "games_played": u32::MAX,
            "total_kills": u32::MAX - 1,
        });
        let stats = merge_settlement(&mut profile, settlement(MatchResult::Defeat)).unwrap();
        assert_eq!(stats.games_played, u32::MAX);
        assert_eq!(stats.wins, 0);
        assert_eq!(stats.total_kills, u32::MAX);
        assert_eq!(profile["total_kp"], 20);
    }

    #[test]
    fn replacement_failure_preserves_original_json() {
        let dir = test_dir("replace-failure");
        let path = dir.join("player_profile.json");
        let original = br#"{"total_kp":20,"games_played":7}"#;
        fs::write(&path, original).unwrap();

        let error = settle_profile_with_replacer(
            &path,
            settlement(MatchResult::Victory),
            |_temporary, _destination| Err(std::io::Error::other("injected replace failure")),
        )
        .unwrap_err();
        assert!(error.contains("result=victory"));
        assert!(error.contains("injected replace failure"));
        assert_eq!(fs::read(&path).unwrap(), original);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn lagging_backend_shutdown_sequence_records_terminal_frontend_once() {
        let dir = test_dir("lagging-backend");
        let path = dir.join("player_profile.json");
        // Represents KP persisted by the backend immediately before the
        // frontend kills/waits it. Backend progress does not own round stats.
        fs::write(&path, r#"{"total_kp":25,"spent_kp":0,"unlocked_nodes":[]}"#).unwrap();
        let mut tracker = MatchSessionTracker::default();
        tracker.start();
        tracker.observe(40, 777);
        tracker.mark_terminal(MatchResult::Victory);

        let terminal = tracker.take_settlement().unwrap();
        let stats = settle_profile(&path, terminal).unwrap();
        assert_eq!(stats.games_played, 1);
        assert_eq!(stats.wins, 1);
        assert_eq!(stats.highest_wave, 40);
        assert_eq!(stats.total_kills, 777);
        assert_eq!(tracker.take_settlement(), None, "teardown is idempotent");
        let saved: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(saved["total_kp"], 25);
        fs::remove_dir_all(dir).ok();
    }
}
