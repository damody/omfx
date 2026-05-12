use std::fs::File;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use tracing_perfetto::PerfettoLayer;
use tracing_subscriber::layer::SubscriberExt;

pub struct PerfettoProfileSession {
    pub path: PathBuf,
    pub detail: String,
}

pub fn init_from_env() -> Result<Option<PerfettoProfileSession>, String> {
    if !perfetto_enabled() {
        return Ok(None);
    }

    let detail = std::env::var("OMFX_PERFETTO_DETAIL").unwrap_or_else(|_| "frame".to_string());
    let path = trace_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|err| {
            format!(
                "failed to create Perfetto trace directory '{}': {}",
                parent.display(),
                err
            )
        })?;
    }

    let file = File::create(&path).map_err(|err| {
        format!(
            "failed to create Perfetto trace file '{}': {}",
            path.display(),
            err
        )
    })?;
    let layer = PerfettoLayer::new(Mutex::new(file))
        .with_debug_annotations(true)
        .with_filter_by_marker(|field_name| field_name == "perfetto");
    let subscriber = tracing_subscriber::registry().with(layer);
    tracing::subscriber::set_global_default(subscriber)
        .map_err(|err| format!("failed to install tracing subscriber: {}", err))?;

    if let Some(seconds) = max_seconds() {
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(seconds));
            log::info!(
                "OMFX_PERFETTO_MAX_SECONDS={} reached; exiting profiling run",
                seconds
            );
            std::process::exit(0);
        });
    }

    Ok(Some(PerfettoProfileSession { path, detail }))
}

fn perfetto_enabled() -> bool {
    std::env::var("OMFX_PERFETTO_TRACE")
        .map(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn trace_path() -> Result<PathBuf, String> {
    if let Ok(path) = std::env::var("OMFX_PERFETTO_PATH") {
        let path = path.trim();
        if !path.is_empty() {
            return Ok(PathBuf::from(path));
        }
    }

    let omfx_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(PathBuf::from)
        .ok_or_else(|| "failed to resolve omfx workspace directory".to_string())?;
    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| format!("system clock before UNIX_EPOCH: {}", err))?
        .as_millis();
    Ok(omfx_dir.join("target").join("profiles").join(format!(
        "omfx-{}-{}.perfetto-trace",
        timestamp_ms,
        std::process::id()
    )))
}

fn max_seconds() -> Option<u64> {
    std::env::var("OMFX_PERFETTO_MAX_SECONDS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
}
