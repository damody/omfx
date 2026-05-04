//! Executor with your game connected to it as a plugin.
use fyrox::engine::executor::Executor;
use fyrox::event_loop::EventLoop;
use fyrox::core::log::Log;
use simplelog::{
    CombinedLogger, ConfigBuilder, LevelFilter, TermLogger, TerminalMode, ColorChoice, WriteLogger,
};
use std::fs::File;

struct LogSettings {
    level: LevelFilter,
    allow_modules: Vec<String>,
}

fn log_level_rank(level: LevelFilter) -> u8 {
    match level {
        LevelFilter::Off => 0,
        LevelFilter::Error => 1,
        LevelFilter::Warn => 2,
        LevelFilter::Info => 3,
        LevelFilter::Debug => 4,
        LevelFilter::Trace => 5,
    }
}

fn parse_log_level(value: &str) -> Option<LevelFilter> {
    match value.to_ascii_lowercase().as_str() {
        "off" => Some(LevelFilter::Off),
        "error" => Some(LevelFilter::Error),
        "warn" => Some(LevelFilter::Warn),
        "info" => Some(LevelFilter::Info),
        "debug" => Some(LevelFilter::Debug),
        "trace" => Some(LevelFilter::Trace),
        _ => None,
    }
}

fn push_allow_module(allow_modules: &mut Vec<String>, module: &str) {
    let module = module.trim();
    if module.is_empty() {
        return;
    }
    let normalized = module.strip_suffix("::lib").unwrap_or(module);
    for candidate in [module, normalized] {
        if !allow_modules.iter().any(|existing| existing == candidate) {
            allow_modules.push(candidate.to_owned());
        }
    }
}

fn configured_log_settings() -> LogSettings {
    let mut settings = LogSettings {
        level: LevelFilter::Info,
        allow_modules: Vec::new(),
    };
    let Ok(value) = std::env::var("RUST_LOG") else { return settings };

    let mut has_bare_level = false;
    // simplelog has no EnvFilter parser. Use the highest requested level and,
    // when only module directives are present, add module allow filters.
    for directive in value.split(',') {
        let directive = directive.trim();
        if directive.is_empty() {
            continue;
        }
        let (candidate, module) = if let Some((module, level)) = directive.split_once('=') {
            let Some(level) = parse_log_level(level.trim()) else { continue };
            (level, Some(module))
        } else {
            let Some(level) = parse_log_level(directive) else { continue };
            has_bare_level = true;
            (level, None)
        };
        if log_level_rank(candidate) > log_level_rank(settings.level) {
            settings.level = candidate;
        }
        if let Some(module) = module {
            push_allow_module(&mut settings.allow_modules, module);
        }
    }
    if has_bare_level {
        settings.allow_modules.clear();
    }
    settings
}

#[cfg(target_os = "windows")]
#[link(name = "winmm")]
extern "system" {
    fn timeBeginPeriod(u_period: u32) -> u32;
}

fn main() {
    // Windows 預設 timer granularity 是 15.6ms，`thread::sleep(Duration::from_millis(1))`
    // 實際會 sleep 8-15ms 把 fps 鎖到 ~60。呼叫 timeBeginPeriod(1) 把全系統 timer
    // resolution 降到 1ms，sleep 就會接近真實 1ms。
    // 配合 fyrox-impl-1.0.1/src/engine/executor.rs 裡 `Event::AboutToWait` 後
    // patch 的 `thread::sleep(Duration::from_millis(1))` 使用，避免 CPU 100% 但又不
    // 過度限速。
    #[cfg(target_os = "windows")]
    unsafe {
        timeBeginPeriod(1);
    }
    // Standard log crate backend：每次啟動 truncate omfx_app.log，同時印到 console。
    // omfx.log 是 fyrox 自己的 log（fyrox::core::log::Log），不走 standard log macros。
    // 我們的 log::info! / warn! 走 simplelog → omfx_app.log + 終端機。
    let log_settings = configured_log_settings();
    let mut cfg_builder = ConfigBuilder::new();
    cfg_builder.set_time_format_rfc3339();
    for module in &log_settings.allow_modules {
        cfg_builder.add_filter_allow(module.clone());
    }
    let cfg = cfg_builder.build();
    let log_file = File::create("omfx_app.log").expect("create omfx_app.log");
    let _ = CombinedLogger::init(vec![
        TermLogger::new(log_settings.level, cfg.clone(), TerminalMode::Mixed, ColorChoice::Auto),
        WriteLogger::new(log_settings.level, cfg, log_file),
    ]);

    Log::set_file_name("omfx.log");

    let mut executor = Executor::from_params(
        EventLoop::new().ok(),
        fyrox::engine::GraphicsContextParams {
            window_attributes: fyrox::window::WindowAttributes::default()
                .with_title("omfx - Tower Defense")
                .with_inner_size(fyrox::dpi::LogicalSize::new(1280.0, 720.0)),
            vsync: false,
            msaa_sample_count: None,
            graphics_server_constructor: Default::default(),
            named_objects: false,
        },
    );

    // Dynamic linking with hot reloading.
    #[cfg(feature = "dylib")]
    {
        #[cfg(target_os = "windows")]
        let file_name = "game_dylib.dll";
        #[cfg(target_os = "linux")]
        let file_name = "libgame_dylib.so";
        #[cfg(target_os = "macos")]
        let file_name = "libgame_dylib.dylib";
        executor.add_dynamic_plugin(file_name, true, true).unwrap();
    }

    // Static linking.
    #[cfg(not(feature = "dylib"))]
    {
        use omfx::Game;
        executor.add_plugin(Game::default());
    }

    executor.run()
}
