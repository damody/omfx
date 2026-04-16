//! Executor with your game connected to it as a plugin.
use fyrox::engine::executor::Executor;
use fyrox::event_loop::EventLoop;
use fyrox::core::log::Log;

fn main() {
    Log::set_file_name("omfx.log");

    let mut executor = Executor::from_params(
        EventLoop::new().ok(),
        fyrox::engine::GraphicsContextParams {
            window_attributes: fyrox::window::WindowAttributes::default()
                .with_title("omfx - Tower Defense"),
            vsync: true,
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