//! 使用 plugin 模式啟動的編輯器。
use fyroxed_base::{fyrox::event_loop::EventLoop, Editor, StartupData, fyrox::core::log::Log};

fn main() {
    Log::set_file_name("omfx.log");

    let event_loop = EventLoop::new().unwrap();
    let mut editor = Editor::new(
        Some(StartupData {
            working_directory: Default::default(),
            scenes: vec!["data/scene.rgs".into()],
            named_objects: false
        }),
    );

    // 動態連結並啟用熱重載。
    #[cfg(feature = "dylib")]
    {
        #[cfg(target_os = "windows")]
        let file_name = "game_dylib.dll";
        #[cfg(target_os = "linux")]
        let file_name = "libgame_dylib.so";
        #[cfg(target_os = "macos")]
        let file_name = "libgame_dylib.dylib";
        editor.add_dynamic_plugin(file_name, true, true).unwrap();
    }

    // 靜態連結。
    #[cfg(not(feature = "dylib"))]
    {
        use omfx::Game;
        editor.add_game_plugin(Game::default());
    }

    editor.run(event_loop)
}
