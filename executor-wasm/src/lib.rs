//! 使用 plugin 模式啟動的網頁執行器。
#![cfg(target_arch = "wasm32")]

use fyrox::core::wasm_bindgen::{self, prelude::*};
use fyrox::engine::{executor::Executor, GraphicsContextParams};
use fyrox::event_loop::EventLoop;
use fyrox::{dpi::LogicalSize, window::WindowAttributes};

use omfx::Game;

#[wasm_bindgen]
pub fn main() {
    let mut executor = Executor::from_params(
        Some(EventLoop::new().unwrap()),
        GraphicsContextParams {
            window_attributes: WindowAttributes::default()
                .with_title("omfx Web")
                .with_resizable(true)
                .with_inner_size(LogicalSize::new(1920.0, 1080.0)),
            vsync: true,
            msaa_sample_count: None,
            graphics_server_constructor: Default::default(),
            named_objects: false,
        },
    );
    executor.add_plugin(Game::default());
    executor.run()
}
