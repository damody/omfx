//! TD 熱鍵設定：動作定義、預設綁定、JSON 存檔、按鍵解析。
//!
//! 設計對齊 BTD6 熱鍵頁：動作分三類（塔選擇 / 遊戲玩法 / 沙箱），
//! 每個動作綁一個（可含 Ctrl modifier 的）按鍵，可重綁、可恢復預設。
//! 存檔格式：`data/hotkeys.json`，內容為 `{ action_id: "Ctrl+Digit1" | "KeyQ" }`。

use fyrox::keyboard::KeyCode;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// 動作定義
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum HotkeyCategory {
    Tower,
    Gameplay,
    Sandbox,
}

pub struct HotkeyDef {
    pub id: &'static str,
    pub label: &'static str,
    pub category: HotkeyCategory,
}

pub const SELECT_TOWER_IDS: [&str; 24] = [
    "select_tower_1",
    "select_tower_2",
    "select_tower_3",
    "select_tower_4",
    "select_tower_5",
    "select_tower_6",
    "select_tower_7",
    "select_tower_8",
    "select_tower_9",
    "select_tower_10",
    "select_tower_11",
    "select_tower_12",
    "select_tower_13",
    "select_tower_14",
    "select_tower_15",
    "select_tower_16",
    "select_tower_17",
    "select_tower_18",
    "select_tower_19",
    "select_tower_20",
    "select_tower_21",
    "select_tower_22",
    "select_tower_23",
    "select_tower_24",
];

pub const SPAWN_CREEP_IDS: [&str; 9] = [
    "spawn_creep_1",
    "spawn_creep_2",
    "spawn_creep_3",
    "spawn_creep_4",
    "spawn_creep_5",
    "spawn_creep_6",
    "spawn_creep_7",
    "spawn_creep_8",
    "spawn_creep_9",
];

/// 目前遊戲內容的固定塔名（依 snapshot.tower_templates 順序）。
/// Doris 拍板：塔名不會變，直接寫死；之後有新塔再補。
pub const KNOWN_TOWER_COUNT: usize = 6;

pub fn hotkey_defs() -> Vec<HotkeyDef> {
    let mut defs = Vec::with_capacity(42);
    const TOWER_LABELS: [&str; 24] = [
        "吉拿棒迫擊砲",
        "飛鏢猴",
        "炸彈射手",
        "鐵釘射手",
        "冰凍猴",
        "蛋糕濺射塔",
        "選塔 7", "選塔 8", "選塔 9",
        "選塔 10", "選塔 11", "選塔 12", "選塔 13", "選塔 14", "選塔 15", "選塔 16", "選塔 17",
        "選塔 18", "選塔 19", "選塔 20", "選塔 21", "選塔 22", "選塔 23", "選塔 24",
    ];
    for (i, id) in SELECT_TOWER_IDS.iter().enumerate() {
        defs.push(HotkeyDef {
            id,
            label: TOWER_LABELS[i],
            category: HotkeyCategory::Tower,
        });
    }
    for (id, label) in [
        ("upgrade_path_1", "升級路線 1"),
        ("upgrade_path_2", "升級路線 2"),
        ("upgrade_path_3", "升級路線 3"),
        ("sell_tower", "賣出"),
        ("change_target", "更改目標"),
        ("change_target_reverse", "反轉更改目標"),
        ("copy_tower", "複製塔"),
        ("start_round", "開始/快進"),
        ("toggle_pause", "暫停"),
    ] {
        defs.push(HotkeyDef {
            id,
            label,
            category: HotkeyCategory::Gameplay,
        });
    }
    const SANDBOX_LABELS: [&str; 9] = [
        "發送敵人 1",
        "發送敵人 2",
        "發送敵人 3",
        "發送敵人 4",
        "發送敵人 5",
        "發送敵人 6",
        "發送敵人 7",
        "發送敵人 8",
        "發送敵人 9",
    ];
    for (i, id) in SPAWN_CREEP_IDS.iter().enumerate() {
        defs.push(HotkeyDef {
            id,
            label: SANDBOX_LABELS[i],
            category: HotkeyCategory::Sandbox,
        });
    }
    defs
}

// ---------------------------------------------------------------------------
// Binding 與 KeyCode 名稱轉換
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Binding {
    pub key: KeyCode,
    pub ctrl: bool,
}

const KEY_NAMES: &[(KeyCode, &str, &str)] = &[
    // (KeyCode, 存檔名, 顯示名)
    (KeyCode::KeyA, "KeyA", "A"),
    (KeyCode::KeyB, "KeyB", "B"),
    (KeyCode::KeyC, "KeyC", "C"),
    (KeyCode::KeyD, "KeyD", "D"),
    (KeyCode::KeyE, "KeyE", "E"),
    (KeyCode::KeyF, "KeyF", "F"),
    (KeyCode::KeyG, "KeyG", "G"),
    (KeyCode::KeyH, "KeyH", "H"),
    (KeyCode::KeyI, "KeyI", "I"),
    (KeyCode::KeyJ, "KeyJ", "J"),
    (KeyCode::KeyK, "KeyK", "K"),
    (KeyCode::KeyL, "KeyL", "L"),
    (KeyCode::KeyM, "KeyM", "M"),
    (KeyCode::KeyN, "KeyN", "N"),
    (KeyCode::KeyO, "KeyO", "O"),
    (KeyCode::KeyP, "KeyP", "P"),
    (KeyCode::KeyQ, "KeyQ", "Q"),
    (KeyCode::KeyR, "KeyR", "R"),
    (KeyCode::KeyS, "KeyS", "S"),
    (KeyCode::KeyT, "KeyT", "T"),
    (KeyCode::KeyU, "KeyU", "U"),
    (KeyCode::KeyV, "KeyV", "V"),
    (KeyCode::KeyW, "KeyW", "W"),
    (KeyCode::KeyX, "KeyX", "X"),
    (KeyCode::KeyY, "KeyY", "Y"),
    (KeyCode::KeyZ, "KeyZ", "Z"),
    (KeyCode::Digit0, "Digit0", "0"),
    (KeyCode::Digit1, "Digit1", "1"),
    (KeyCode::Digit2, "Digit2", "2"),
    (KeyCode::Digit3, "Digit3", "3"),
    (KeyCode::Digit4, "Digit4", "4"),
    (KeyCode::Digit5, "Digit5", "5"),
    (KeyCode::Digit6, "Digit6", "6"),
    (KeyCode::Digit7, "Digit7", "7"),
    (KeyCode::Digit8, "Digit8", "8"),
    (KeyCode::Digit9, "Digit9", "9"),
    (KeyCode::Comma, "Comma", ","),
    (KeyCode::Period, "Period", "."),
    (KeyCode::Slash, "Slash", "/"),
    (KeyCode::Semicolon, "Semicolon", ";"),
    (KeyCode::Quote, "Quote", "'"),
    (KeyCode::BracketLeft, "BracketLeft", "["),
    (KeyCode::BracketRight, "BracketRight", "]"),
    (KeyCode::Backslash, "Backslash", "\\"),
    (KeyCode::Minus, "Minus", "-"),
    (KeyCode::Equal, "Equal", "="),
    (KeyCode::Backquote, "Backquote", "`"),
    (KeyCode::Space, "Space", "Space"),
    (KeyCode::Backspace, "Backspace", "Backspace"),
    (KeyCode::Tab, "Tab", "Tab"),
    (KeyCode::Enter, "Enter", "Enter"),
    (KeyCode::Insert, "Insert", "Insert"),
    (KeyCode::Delete, "Delete", "Delete"),
    (KeyCode::Home, "Home", "Home"),
    (KeyCode::End, "End", "End"),
    (KeyCode::PageUp, "PageUp", "PageUp"),
    (KeyCode::PageDown, "PageDown", "PageDown"),
];

fn key_to_name(key: KeyCode) -> Option<&'static str> {
    KEY_NAMES.iter().find(|(k, _, _)| *k == key).map(|(_, n, _)| *n)
}

fn key_from_name(name: &str) -> Option<KeyCode> {
    KEY_NAMES.iter().find(|(_, n, _)| *n == name).map(|(k, _, _)| *k)
}

pub fn key_display(key: KeyCode) -> &'static str {
    KEY_NAMES
        .iter()
        .find(|(k, _, _)| *k == key)
        .map(|(_, _, d)| *d)
        .unwrap_or("?")
}

/// 熱鍵是否為可綁定的按鍵（modifier 鍵與 Escape/F 鍵不可綁）。
pub fn is_bindable_key(key: KeyCode) -> bool {
    key_to_name(key).is_some()
}

impl Binding {
    fn to_saved(self) -> String {
        let name = key_to_name(self.key).unwrap_or("?");
        if self.ctrl {
            format!("Ctrl+{}", name)
        } else {
            name.to_string()
        }
    }

    fn from_saved(raw: &str) -> Option<Self> {
        let (ctrl, name) = match raw.strip_prefix("Ctrl+") {
            Some(rest) => (true, rest),
            None => (false, raw),
        };
        Some(Binding {
            key: key_from_name(name)?,
            ctrl,
        })
    }

    pub fn display(self) -> String {
        if self.ctrl {
            format!("Ctrl+{}", key_display(self.key))
        } else {
            key_display(self.key).to_string()
        }
    }
}

// ---------------------------------------------------------------------------
// HotkeyConfig
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct HotkeyConfig {
    /// action_id → binding。未綁定的動作不在 map 裡。
    pub bindings: HashMap<String, Binding>,
}

impl Default for HotkeyConfig {
    fn default() -> Self {
        Self::defaults()
    }
}

const CONFIG_REL_PATH: &str = "data/hotkeys.json";

fn config_candidate_paths() -> Vec<String> {
    // 與 load_sound_from_path 相同的 CWD 佈局候選。
    vec![
        CONFIG_REL_PATH.to_string(),
        format!("omfx/game/{}", CONFIG_REL_PATH),
        format!("game/{}", CONFIG_REL_PATH),
        format!("../{}", CONFIG_REL_PATH),
    ]
}

impl HotkeyConfig {
    pub fn defaults() -> Self {
        let mut bindings = HashMap::new();
        // 塔選擇：完整 BTD6 順序
        const TOWER_KEYS: [KeyCode; 24] = [
            KeyCode::KeyQ,
            KeyCode::KeyW,
            KeyCode::KeyE,
            KeyCode::KeyR,
            KeyCode::KeyT,
            KeyCode::KeyY,
            KeyCode::KeyZ,
            KeyCode::KeyX,
            KeyCode::KeyC,
            KeyCode::KeyV,
            KeyCode::KeyB,
            KeyCode::KeyN,
            KeyCode::KeyM,
            KeyCode::KeyA,
            KeyCode::KeyS,
            KeyCode::KeyD,
            KeyCode::KeyF,
            KeyCode::KeyG,
            KeyCode::KeyH,
            KeyCode::KeyJ,
            KeyCode::KeyK,
            KeyCode::KeyL,
            KeyCode::KeyI,
            KeyCode::KeyU,
        ];
        for (i, id) in SELECT_TOWER_IDS.iter().enumerate() {
            bindings.insert(
                id.to_string(),
                Binding {
                    key: TOWER_KEYS[i],
                    ctrl: false,
                },
            );
        }
        for (id, key, ctrl) in [
            ("upgrade_path_1", KeyCode::Comma, false),
            ("upgrade_path_2", KeyCode::Period, false),
            ("upgrade_path_3", KeyCode::Slash, false),
            ("sell_tower", KeyCode::Backspace, false),
            ("change_target", KeyCode::Tab, false),
            ("change_target_reverse", KeyCode::Tab, true),
            ("copy_tower", KeyCode::KeyC, true),
            ("start_round", KeyCode::Space, false),
            ("toggle_pause", KeyCode::Backquote, false),
        ] {
            bindings.insert(id.to_string(), Binding { key, ctrl });
        }
        const SPAWN_KEYS: [KeyCode; 9] = [
            KeyCode::Digit1,
            KeyCode::Digit2,
            KeyCode::Digit3,
            KeyCode::Digit4,
            KeyCode::Digit5,
            KeyCode::Digit6,
            KeyCode::Digit7,
            KeyCode::Digit8,
            KeyCode::Digit9,
        ];
        for (i, id) in SPAWN_CREEP_IDS.iter().enumerate() {
            bindings.insert(
                id.to_string(),
                Binding {
                    key: SPAWN_KEYS[i],
                    ctrl: true,
                },
            );
        }
        Self { bindings }
    }

    /// 讀取存檔並套在預設值之上；沒有存檔就用預設。
    pub fn load() -> Self {
        let mut cfg = Self::defaults();
        for path in config_candidate_paths() {
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            match serde_json::from_str::<HashMap<String, String>>(&text) {
                Ok(saved) => {
                    for (id, raw) in saved {
                        if let Some(b) = Binding::from_saved(&raw) {
                            cfg.bindings.insert(id, b);
                        }
                    }
                    log::info!("Hotkeys loaded from: {}", path);
                }
                Err(e) => log::warn!("hotkeys.json 解析失敗（{}）：{}", path, e),
            }
            break;
        }
        cfg
    }

    pub fn save(&self) {
        let saved: HashMap<&str, String> = self
            .bindings
            .iter()
            .map(|(id, b)| (id.as_str(), b.to_saved()))
            .collect();
        let Ok(json) = serde_json::to_string_pretty(&saved) else {
            return;
        };
        for path in config_candidate_paths() {
            let parent_ok = std::path::Path::new(&path)
                .parent()
                .map(|p| p.is_dir())
                .unwrap_or(false);
            if !parent_ok {
                continue;
            }
            match std::fs::write(&path, &json) {
                Ok(()) => {
                    log::info!("Hotkeys saved to: {}", path);
                    return;
                }
                Err(e) => log::warn!("hotkeys.json 寫入失敗（{}）：{}", path, e),
            }
        }
        log::warn!("hotkeys.json 找不到可寫入的 data/ 目錄");
    }

    /// 解析按鍵 → action_id。Ctrl 綁定優先精確匹配。
    pub fn resolve(&self, key: KeyCode, ctrl: bool) -> Option<&str> {
        self.bindings
            .iter()
            .find(|(_, b)| b.key == key && b.ctrl == ctrl)
            .map(|(id, _)| id.as_str())
    }

    pub fn binding_display(&self, id: &str) -> String {
        self.bindings
            .get(id)
            .map(|b| b.display())
            .unwrap_or_else(|| "—".to_string())
    }

    /// 重綁：若新按鍵已被其他動作占用，兩者互換綁定（BTD6 行為）。
    pub fn rebind(&mut self, id: &str, key: KeyCode, ctrl: bool) {
        let new_b = Binding { key, ctrl };
        let conflict = self
            .bindings
            .iter()
            .find(|(other, b)| other.as_str() != id && **b == new_b)
            .map(|(other, _)| other.clone());
        if let Some(other) = conflict {
            if let Some(old) = self.bindings.get(id).copied() {
                self.bindings.insert(other, old);
            } else {
                self.bindings.remove(&other);
            }
        }
        self.bindings.insert(id.to_string(), new_b);
    }
}
