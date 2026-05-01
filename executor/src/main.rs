//! Executor with your game connected to it as a plugin.
use fyrox::engine::executor::Executor;
use fyrox::event_loop::EventLoop;
use fyrox::core::log::Log;
use simplelog::{
    CombinedLogger, ConfigBuilder, LevelFilter, TermLogger, TerminalMode, ColorChoice, WriteLogger,
};
use std::fs::File;

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
    let cfg = ConfigBuilder::new()
        .set_time_format_rfc3339()
        .build();
    let log_file = File::create("omfx_app.log").expect("create omfx_app.log");
    let _ = CombinedLogger::init(vec![
        TermLogger::new(LevelFilter::Info, cfg.clone(), TerminalMode::Mixed, ColorChoice::Auto),
        WriteLogger::new(LevelFilter::Info, cfg, log_file),
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