//! 匯出指令工具（CLI），會以 plugin 方式啟動遊戲後輸出專案資料。
//! 此工具可用於 CI/CD 自動化專案匯出流程。
//! 範例：`cargo run --package export-cli -- --target-platform pc`
//!       或 `cargo run --package export-cli -- --help` 查閱說明。

use omfx::Game;
use fyrox::core::log::Log;
use fyrox::engine::executor::Executor;
use fyrox::event_loop::EventLoop;
use fyrox_build_tools::export::cli_export;

fn main() {
    Log::set_file_name("omfxExport.log");
    let mut executor = Executor::new(EventLoop::new().ok());
    executor.add_plugin(Game::default());
    cli_export(executor.resource_manager.clone())
}
