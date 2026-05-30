use std::path::PathBuf;
use std::process::{Child, Command};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackendLaunchConfig {
    pub session_id: String,
    pub map_id: String,
    pub story: String,
    pub difficulty_id: String,
    pub difficulty_config: String,
    pub kcp_addr: String,
    pub content_root: Option<PathBuf>,
    pub executable: Option<PathBuf>,
    pub launcher_enabled: bool,
}

impl BackendLaunchConfig {
    pub fn launch_env(&self) -> Vec<(String, String)> {
        let mut envs = vec![
            ("OMB_SESSION_ID".into(), self.session_id.clone()),
            ("OMB_MAP_ID".into(), self.map_id.clone()),
            ("OMB_STORY".into(), self.story.clone()),
            ("OMB_DIFFICULTY_ID".into(), self.difficulty_id.clone()),
            ("OMB_DIFFICULTY".into(), self.difficulty_config.clone()),
            ("OMB_KCP_ADDR".into(), self.kcp_addr.clone()),
        ];
        if let Some(root) = &self.content_root {
            let root = absolute_path(root);
            envs.push((
                "OMB_CONTENT_ROOT".into(),
                root.to_string_lossy().into_owned(),
            ));
            envs.push((
                "OMB_LUA_CONTENT_ROOT".into(),
                root.to_string_lossy().into_owned(),
            ));
        }
        for key in [
            "OMB_GAME_TOML",
            "OMB_DLL_PATH",
            "OMB_SCRIPTS_DIR",
            "OMB_STORY_DATA_DIR",
            "OMB_LUA_CONTENT_ROOT",
            "OMB_LUA_CONTENT",
            "OMB_LUA_HOT_RELOAD",
        ] {
            if let Ok(value) = std::env::var(key) {
                let value = if matches!(
                    key,
                    "OMB_GAME_TOML"
                        | "OMB_DLL_PATH"
                        | "OMB_SCRIPTS_DIR"
                        | "OMB_STORY_DATA_DIR"
                        | "OMB_LUA_CONTENT_ROOT"
                ) {
                    absolute_path(PathBuf::from(value))
                        .to_string_lossy()
                        .into_owned()
                } else {
                    value
                };
                envs.push((key.into(), value));
            }
        }
        envs
    }
}

#[derive(Debug)]
pub struct BackendSession {
    config: BackendLaunchConfig,
    child: Option<Child>,
    shutdown: bool,
}

impl BackendSession {
    pub fn start(config: BackendLaunchConfig) -> Result<Self, String> {
        if !config.launcher_enabled {
            log::info!(
                "backend-session external mode: session_id={} map_id={} story={} difficulty={} addr={}",
                config.session_id,
                config.map_id,
                config.story,
                config.difficulty_id,
                config.kcp_addr
            );
            return Ok(Self {
                config,
                child: None,
                shutdown: false,
            });
        }

        let executable = config
            .executable
            .clone()
            .or_else(find_backend_executable)
            .ok_or_else(|| "backend executable not found; set OMFX_BACKEND_EXE".to_string())?;
        let mut command = Command::new(&executable);
        command.current_dir(backend_working_dir(&executable));
        for (key, value) in config.launch_env() {
            command.env(key, value);
        }
        log::info!(
            "backend-session start: exe={:?} session_id={} map_id={} story={} difficulty={} addr={}",
            executable,
            config.session_id,
            config.map_id,
            config.story,
            config.difficulty_id,
            config.kcp_addr
        );
        let child = command
            .spawn()
            .map_err(|err| format!("failed to start backend {:?}: {}", executable, err))?;
        Ok(Self {
            config,
            child: Some(child),
            shutdown: false,
        })
    }

    pub fn owns_process(&self) -> bool {
        self.child.is_some()
    }

    pub fn addr(&self) -> &str {
        &self.config.kcp_addr
    }

    pub fn shutdown(&mut self) {
        if self.shutdown {
            return;
        }
        self.shutdown = true;
        if let Some(mut child) = self.child.take() {
            log::info!(
                "backend-session shutdown: session_id={} map_id={} story={} difficulty={}",
                self.config.session_id,
                self.config.map_id,
                self.config.story,
                self.config.difficulty_id
            );
            if let Err(err) = child.kill() {
                log::warn!("backend-session kill failed: {}", err);
            }
            if let Err(err) = child.wait() {
                log::warn!("backend-session wait failed: {}", err);
            }
        }
    }
}

impl Drop for BackendSession {
    fn drop(&mut self) {
        self.shutdown();
    }
}

pub fn launcher_enabled_from_env() -> bool {
    !env_truthy("OMFX_EXTERNAL_BACKEND") && !env_truthy("OMFX_DISABLE_SESSION_LAUNCHER")
}

pub fn configured_backend_executable() -> Option<PathBuf> {
    std::env::var("OMFX_BACKEND_EXE")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from)
}

fn find_backend_executable() -> Option<PathBuf> {
    if let Some(path) = configured_backend_executable() {
        return Some(path);
    }
    let exe_name = if cfg!(windows) {
        "omobab.exe"
    } else {
        "omobab"
    };
    let mut candidates = Vec::new();
    if let Ok(current) = std::env::current_exe() {
        if let Some(dir) = current.parent() {
            candidates.push(dir.join(exe_name));
            candidates.push(dir.join("../backend").join(exe_name));
            candidates.push(dir.join("../../omb/target/debug").join(exe_name));
            candidates.push(dir.join("../../omb/target/release").join(exe_name));
            candidates.push(dir.join("../../../omb/target/debug").join(exe_name));
            candidates.push(dir.join("../../../omb/target/release").join(exe_name));
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join("../omb/target/debug").join(exe_name));
        candidates.push(cwd.join("../omb/target/release").join(exe_name));
        candidates.push(cwd.join("omb/target/debug").join(exe_name));
        candidates.push(cwd.join("omb/target/release").join(exe_name));
    }
    candidates.push(PathBuf::from("target/debug").join(exe_name));
    candidates.push(PathBuf::from("target/release").join(exe_name));
    candidates.into_iter().find(|path| path.exists())
}

fn backend_working_dir(executable: &std::path::Path) -> PathBuf {
    let Some(parent) = executable.parent() else {
        return PathBuf::from(".");
    };
    let Some(profile_dir) = parent.file_name().and_then(|name| name.to_str()) else {
        return parent.to_path_buf();
    };
    if matches!(profile_dir, "debug" | "release") {
        if let Some(target_dir) = parent.parent() {
            if target_dir.file_name().and_then(|name| name.to_str()) == Some("target") {
                if let Some(crate_dir) = target_dir.parent() {
                    return crate_dir.to_path_buf();
                }
            }
        }
    }
    parent.to_path_buf()
}

fn env_truthy(key: &str) -> bool {
    std::env::var(key)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn absolute_path(path: impl AsRef<std::path::Path>) -> PathBuf {
    let path = path.as_ref();
    if path.is_absolute() {
        return path.to_path_buf();
    }
    std::env::current_dir()
        .map(|cwd| cwd.join(path))
        .unwrap_or_else(|_| path.to_path_buf())
        .canonicalize()
        .unwrap_or_else(|_| {
            std::env::current_dir()
                .map(|cwd| cwd.join(path))
                .unwrap_or_else(|_| path.to_path_buf())
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn external_backend_session_is_not_owned_by_launcher() {
        let config = BackendLaunchConfig {
            session_id: "session-1".into(),
            map_id: "map".into(),
            story: "TD_1".into(),
            difficulty_id: "easy".into(),
            difficulty_config: "easy".into(),
            kcp_addr: "127.0.0.1:50061".into(),
            content_root: None,
            executable: None,
            launcher_enabled: false,
        };

        let session = BackendSession::start(config).expect("external session starts");

        assert!(!session.owns_process());
        assert_eq!(session.addr(), "127.0.0.1:50061");
    }

    #[test]
    fn external_backend_shutdown_is_idempotent() {
        let config = BackendLaunchConfig {
            session_id: "session-shutdown".into(),
            map_id: "map".into(),
            story: "TD_1".into(),
            difficulty_id: "easy".into(),
            difficulty_config: "easy".into(),
            kcp_addr: "127.0.0.1:50061".into(),
            content_root: None,
            executable: None,
            launcher_enabled: false,
        };
        let mut session = BackendSession::start(config).expect("external session starts");

        session.shutdown();
        session.shutdown();

        assert!(!session.owns_process());
    }

    #[test]
    fn launch_command_contains_session_metadata() {
        let config = BackendLaunchConfig {
            session_id: "session-2".into(),
            map_id: "map_b".into(),
            story: "TD_2".into(),
            difficulty_id: "hard".into(),
            difficulty_config: "hard_plus".into(),
            kcp_addr: "127.0.0.1:50100".into(),
            content_root: Some(std::path::PathBuf::from("scripts/base_content")),
            executable: Some(std::path::PathBuf::from("omobab.exe")),
            launcher_enabled: true,
        };

        let envs = config.launch_env();

        assert!(envs.contains(&("OMB_SESSION_ID".into(), "session-2".into())));
        assert!(envs.contains(&("OMB_MAP_ID".into(), "map_b".into())));
        assert!(envs.contains(&("OMB_STORY".into(), "TD_2".into())));
        assert!(envs.contains(&("OMB_DIFFICULTY".into(), "hard_plus".into())));
        assert!(envs.contains(&("OMB_DIFFICULTY_ID".into(), "hard".into())));
        assert!(envs.contains(&("OMB_KCP_ADDR".into(), "127.0.0.1:50100".into())));
    }

    #[test]
    fn dev_backend_binary_uses_crate_dir_as_working_dir() {
        let path = std::path::Path::new("/repo/omb/target/debug/omobab");

        assert_eq!(
            backend_working_dir(path),
            std::path::PathBuf::from("/repo/omb")
        );
    }

    #[test]
    fn launch_command_sets_lua_content_root_for_backend() {
        let config = BackendLaunchConfig {
            session_id: "session-3".into(),
            map_id: "map_c".into(),
            story: "TD_1".into(),
            difficulty_id: "easy".into(),
            difficulty_config: "easy".into(),
            kcp_addr: "127.0.0.1:50061".into(),
            content_root: Some(std::path::PathBuf::from("scripts/lua_data")),
            executable: None,
            launcher_enabled: true,
        };

        let envs = config.launch_env();

        assert!(envs.iter().any(|(key, _)| key == "OMB_CONTENT_ROOT"));
        assert!(envs.iter().any(
            |(key, value)| key == "OMB_LUA_CONTENT_ROOT" && value.ends_with("scripts/lua_data")
        ));
    }
}
