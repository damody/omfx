//! 使用 plugin 模式啟動的網頁執行器。
#![cfg(target_arch = "wasm32")]

use fyrox::engine::executor::Executor;
use fyrox::event_loop::EventLoop;
use fyrox::core::wasm_bindgen::{self, prelude::*};

use omfx::Game;

#[wasm_bindgen]
pub fn main() {
    let mut executor = Executor::new(Some(EventLoop::new().unwrap()));
    executor.add_plugin(Game::default());
    executor.run()
}
