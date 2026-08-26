use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use omoba_core::runtime::{
    run_td_autoplay_1_to_100_observed, SimWorldSnapshot, TdAutoplayFrame,
    TdAutoplayObservedOutcome, TdAutoplayObserverControl, TdAutoplayRunConfig, TdAutoplayRunStatus,
};

pub const VISUAL_AUTOPLAY_PUBLISH_INTERVAL: Duration = Duration::from_millis(100);

type LatestFrame = Arc<Mutex<Option<Arc<TdAutoplayFrame>>>>;

#[derive(Debug)]
pub struct VisualAutoplayHandle {
    latest: LatestFrame,
    pub state: Arc<Mutex<SimWorldSnapshot>>,
    cancel: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl VisualAutoplayHandle {
    pub fn spawn(scripts_dir: PathBuf) -> Self {
        let latest = LatestFrame::default();
        let latest_for_thread = latest.clone();
        let state = Arc::new(Mutex::new(SimWorldSnapshot::default()));
        let state_for_thread = state.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel_for_thread = cancel.clone();
        let thread = thread::Builder::new()
            .name("omfx-visual-autoplay".to_string())
            .spawn(move || {
                let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let config = TdAutoplayRunConfig::coarse_1_to_100(scripts_dir);
                    run_td_autoplay_1_to_100_observed(
                        &config,
                        VISUAL_AUTOPLAY_PUBLISH_INTERVAL,
                        |frame| {
                            if let Ok(mut snapshot) = state_for_thread.lock() {
                                *snapshot = frame.snapshot.clone();
                            }
                            publish_latest(&latest_for_thread, frame.clone());
                            if cancel_for_thread.load(Ordering::Acquire) {
                                TdAutoplayObserverControl::Cancel
                            } else {
                                TdAutoplayObserverControl::Continue
                            }
                        },
                    )
                }));
                match run {
                    Ok(Ok(TdAutoplayObservedOutcome::Completed(report))) => {
                        log::info!("visual autoplay completed: {}", report.compact_summary());
                    }
                    Ok(Ok(TdAutoplayObservedOutcome::Cancelled)) => {
                        log::info!("visual autoplay cancelled");
                    }
                    Ok(Err(error)) => {
                        log::error!("visual autoplay failed: {error}");
                        publish_terminal_failure(&latest_for_thread, error);
                    }
                    Err(_) => {
                        let error = "visual autoplay worker panicked".to_string();
                        log::error!("{error}");
                        publish_terminal_failure(&latest_for_thread, error);
                    }
                }
            })
            .expect("spawn omfx visual autoplay worker");
        Self {
            latest,
            state,
            cancel,
            thread: Some(thread),
        }
    }

    pub fn latest_frame(&self) -> Option<Arc<TdAutoplayFrame>> {
        self.latest.lock().ok()?.clone()
    }

    pub fn state_if_ready(&self) -> Option<Arc<Mutex<SimWorldSnapshot>>> {
        self.latest_frame()?;
        Some(self.state.clone())
    }

    pub fn request_cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }
}

impl Drop for VisualAutoplayHandle {
    fn drop(&mut self) {
        self.request_cancel();
        if let Some(thread) = self.thread.take() {
            if thread.join().is_err() {
                log::error!("visual autoplay worker panicked during shutdown");
            }
        }
    }
}

fn publish_latest(latest: &LatestFrame, frame: TdAutoplayFrame) {
    if let Ok(mut slot) = latest.lock() {
        *slot = Some(Arc::new(frame));
    }
    thread::yield_now();
}

fn publish_terminal_failure(latest: &LatestFrame, error: String) {
    let frame = latest
        .lock()
        .ok()
        .and_then(|slot| slot.clone())
        .map(|frame| (*frame).clone())
        .unwrap_or_else(|| TdAutoplayFrame {
            snapshot: SimWorldSnapshot::default(),
            status: TdAutoplayRunStatus::Failed,
            round: 1,
            total_rounds: 100,
            completion_percent: 1,
            tick: 0,
            cash: 0,
            lives: 0,
            tower_count: 0,
            enemy_count: 0,
            error: None,
        });
    publish_latest(
        latest,
        TdAutoplayFrame {
            status: TdAutoplayRunStatus::Failed,
            error: Some(error),
            ..frame
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(tick: u64, status: TdAutoplayRunStatus) -> TdAutoplayFrame {
        TdAutoplayFrame {
            snapshot: SimWorldSnapshot {
                tick: tick as u32,
                ..Default::default()
            },
            status,
            round: 1,
            total_rounds: 100,
            completion_percent: 1,
            tick,
            cash: 650,
            lives: 100,
            tower_count: 1,
            enemy_count: 0,
            error: None,
        }
    }

    #[test]
    fn latest_frame_overwrites_without_backlog() {
        let latest = LatestFrame::default();
        publish_latest(&latest, frame(1, TdAutoplayRunStatus::Running));
        publish_latest(&latest, frame(20, TdAutoplayRunStatus::Running));
        let stored = latest.lock().unwrap().clone().unwrap();
        assert_eq!(stored.tick, 20);
    }

    #[test]
    fn terminal_failure_preserves_last_snapshot_and_updates_status() {
        let latest = LatestFrame::default();
        publish_latest(&latest, frame(9, TdAutoplayRunStatus::Running));
        publish_terminal_failure(&latest, "boom".to_string());
        let stored = latest.lock().unwrap().clone().unwrap();
        assert_eq!(stored.tick, 9);
        assert_eq!(stored.status, TdAutoplayRunStatus::Failed);
        assert_eq!(stored.error.as_deref(), Some("boom"));
    }

    #[test]
    fn terminal_failure_can_publish_before_world_initialization() {
        let latest = LatestFrame::default();
        publish_terminal_failure(&latest, "missing scripts".to_string());
        let stored = latest.lock().unwrap().clone().unwrap();
        assert_eq!(stored.status, TdAutoplayRunStatus::Failed);
        assert_eq!(stored.total_rounds, 100);
        assert_eq!(stored.error.as_deref(), Some("missing scripts"));
    }
}
