use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PregameState {
    MainMenu,
    MapSelect,
    DifficultySelect,
    StartingSession,
    InGame,
    SessionEnded,
}

impl Default for PregameState {
    fn default() -> Self {
        Self::MainMenu
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PregameRuntime {
    pub state: PregameState,
    pub catalog: PregameCatalog,
    pub selected_map: Option<MapEntry>,
    pub selected_difficulty: Option<DifficultyEntry>,
    pub has_gameplay_session: bool,
    pub last_error: Option<String>,
}

impl PregameRuntime {
    pub fn new_for_menu(catalog: PregameCatalog) -> Self {
        Self {
            state: PregameState::MainMenu,
            catalog,
            selected_map: None,
            selected_difficulty: None,
            has_gameplay_session: false,
            last_error: None,
        }
    }

    pub fn active_screen_id(&self) -> &'static str {
        match self.state {
            PregameState::MainMenu => "main_menu",
            PregameState::MapSelect => "map_select",
            PregameState::DifficultySelect => "difficulty_select",
            PregameState::StartingSession => "starting_session",
            PregameState::InGame => "in_game",
            PregameState::SessionEnded => "session_ended",
        }
    }

    pub fn is_pregame(&self) -> bool {
        !matches!(self.state, PregameState::InGame)
    }

    pub fn dispatch(&mut self, action: &PregameAction) -> Option<SessionSelection> {
        match action {
            PregameAction::Navigate { target } if target == "map_select" => {
                self.selected_map = None;
                self.selected_difficulty = None;
                self.state = PregameState::MapSelect;
                None
            }
            PregameAction::Navigate { target } if target == "main_menu" => {
                self.selected_map = None;
                self.selected_difficulty = None;
                self.state = PregameState::MainMenu;
                None
            }
            PregameAction::Back => {
                match self.state {
                    PregameState::MapSelect => {
                        self.selected_map = None;
                        self.selected_difficulty = None;
                        self.state = PregameState::MainMenu;
                    }
                    PregameState::DifficultySelect => {
                        self.selected_difficulty = None;
                        self.state = PregameState::MapSelect;
                    }
                    PregameState::SessionEnded => {
                        self.state = PregameState::MainMenu;
                    }
                    _ => {}
                }
                None
            }
            PregameAction::SelectMap { map_id } => {
                let Some(map) = self.catalog.map(map_id) else {
                    return None;
                };
                if !map.is_playable() {
                    return None;
                }
                self.selected_map = Some(map.clone());
                self.selected_difficulty = None;
                self.state = PregameState::DifficultySelect;
                None
            }
            PregameAction::SelectDifficulty { difficulty_id } => {
                let Some(difficulty) = self.catalog.difficulty(difficulty_id) else {
                    return None;
                };
                if !difficulty.enabled {
                    return None;
                }
                self.selected_difficulty = Some(difficulty.clone());
                self.start_selection()
            }
            PregameAction::StartSession => self.start_selection(),
            PregameAction::NoOp | PregameAction::Navigate { .. } => None,
        }
    }

    pub fn start_selection(&mut self) -> Option<SessionSelection> {
        let (Some(map), Some(difficulty)) =
            (self.selected_map.clone(), self.selected_difficulty.clone())
        else {
            return None;
        };
        if !map.is_playable() || !difficulty.enabled {
            return None;
        }
        self.state = PregameState::StartingSession;
        Some(SessionSelection { map, difficulty })
    }

    pub fn mark_in_game(&mut self) {
        self.has_gameplay_session = true;
        self.state = PregameState::InGame;
        self.last_error = None;
    }

    pub fn recover_to_difficulty(&mut self, error: impl Into<String>) {
        self.has_gameplay_session = false;
        self.state = PregameState::DifficultySelect;
        self.last_error = Some(error.into());
    }

    pub fn return_to_menu(&mut self) {
        self.has_gameplay_session = false;
        self.selected_map = None;
        self.selected_difficulty = None;
        self.state = PregameState::MainMenu;
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SessionSelection {
    pub map: MapEntry,
    pub difficulty: DifficultyEntry,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PregameCatalog {
    pub screens: Vec<ScreenEntry>,
    pub maps: Vec<MapEntry>,
    pub difficulties: Vec<DifficultyEntry>,
    pub diagnostics: Vec<String>,
    pub source_path: Option<PathBuf>,
    pub used_fallback: bool,
}

impl PregameCatalog {
    pub fn load() -> Self {
        let mut diagnostics = Vec::new();
        for path in candidate_catalog_paths() {
            match std::fs::read_to_string(&path) {
                Ok(text) => match Self::from_json_str(&text) {
                    Ok(mut catalog) => {
                        catalog.source_path = Some(path);
                        catalog.validate_asset_paths();
                        return catalog;
                    }
                    Err(err) => {
                        diagnostics.push(format!("pregame catalog {:?} invalid: {}", path, err))
                    }
                },
                Err(err) => {
                    diagnostics.push(format!("pregame catalog {:?} unavailable: {}", path, err));
                }
            }
        }

        let mut fallback = Self::fallback();
        fallback.used_fallback = true;
        fallback
            .diagnostics
            .push("using frontend fallback pregame catalog".to_string());
        fallback.diagnostics.extend(diagnostics);
        fallback
    }

    pub fn from_json_str(text: &str) -> Result<Self, serde_json::Error> {
        let raw: RawPregameCatalog = serde_json::from_str(text)?;
        let mut diagnostics = Vec::new();
        let screens = raw
            .screens
            .into_iter()
            .map(|screen| ScreenEntry {
                id: screen.id,
                title: screen.title,
                subtitle: screen.subtitle.unwrap_or_default(),
                background_image: screen.background_image,
                widgets: screen
                    .widgets
                    .into_iter()
                    .map(|widget| {
                        let id = widget.id;
                        let action = parse_action(widget.action, &id, &mut diagnostics);
                        WidgetEntry {
                            id,
                            label: widget.label,
                            description: widget.description.unwrap_or_default(),
                            image: widget.image,
                            enabled: widget.enabled.unwrap_or(true),
                            locked: widget.locked.unwrap_or(false),
                            action,
                        }
                    })
                    .collect(),
            })
            .collect();
        let maps = raw
            .maps
            .into_iter()
            .map(|map| MapEntry {
                id: map.id,
                label: map.label,
                description: map.description.unwrap_or_default(),
                story: map.story.unwrap_or_default(),
                runtime: map.runtime.unwrap_or_default(),
                image: map.image,
                enabled: map.enabled.unwrap_or(true),
                locked: map.locked.unwrap_or(false),
                reward: map.reward.unwrap_or_default(),
            })
            .collect();
        let difficulties = raw
            .difficulties
            .into_iter()
            .map(|difficulty| DifficultyEntry {
                id: difficulty.id,
                label: difficulty.label,
                description: difficulty.description.unwrap_or_default(),
                config: difficulty.config.unwrap_or_default(),
                reward: difficulty.reward.unwrap_or_default(),
                image: difficulty.image,
                enabled: difficulty.enabled.unwrap_or(true),
            })
            .collect();
        let mut catalog = Self {
            screens,
            maps,
            difficulties,
            diagnostics,
            source_path: None,
            used_fallback: false,
        };
        catalog.validate_required_session_data();
        Ok(catalog)
    }

    pub fn fallback() -> Self {
        Self {
            screens: vec![
                ScreenEntry {
                    id: "main_menu".into(),
                    title: "Open MOBA TD".into(),
                    subtitle: "Choose a map and difficulty to start".into(),
                    background_image: None,
                    widgets: vec![
                        WidgetEntry {
                            id: "start".into(),
                            label: "Start".into(),
                            description: String::new(),
                            image: None,
                            enabled: true,
                            locked: false,
                            action: PregameAction::Navigate {
                                target: "map_select".into(),
                            },
                        },
                        WidgetEntry {
                            id: "settings".into(),
                            label: "Settings".into(),
                            description: "Coming soon".into(),
                            image: None,
                            enabled: false,
                            locked: true,
                            action: PregameAction::NoOp,
                        },
                    ],
                },
                ScreenEntry {
                    id: "map_select".into(),
                    title: "Select Map".into(),
                    subtitle: String::new(),
                    background_image: None,
                    widgets: vec![WidgetEntry {
                        id: "back".into(),
                        label: "Back".into(),
                        description: String::new(),
                        image: None,
                        enabled: true,
                        locked: false,
                        action: PregameAction::Back,
                    }],
                },
                ScreenEntry {
                    id: "difficulty_select".into(),
                    title: "Select Difficulty".into(),
                    subtitle: String::new(),
                    background_image: None,
                    widgets: vec![WidgetEntry {
                        id: "back".into(),
                        label: "Back".into(),
                        description: String::new(),
                        image: None,
                        enabled: true,
                        locked: false,
                        action: PregameAction::Back,
                    }],
                },
            ],
            maps: vec![MapEntry {
                id: "td_1".into(),
                label: "Green Crossing".into(),
                description: "Classic TD lane defense".into(),
                story: "TD_1".into(),
                runtime: "TD_1".into(),
                image: None,
                enabled: true,
                locked: false,
                reward: "100 gold".into(),
            }],
            difficulties: vec![
                DifficultyEntry {
                    id: "easy".into(),
                    label: "Easy".into(),
                    description: "Relaxed waves".into(),
                    config: "easy".into(),
                    reward: "1x".into(),
                    image: None,
                    enabled: true,
                },
                DifficultyEntry {
                    id: "medium".into(),
                    label: "Medium".into(),
                    description: "Standard challenge".into(),
                    config: "medium".into(),
                    reward: "1.25x".into(),
                    image: None,
                    enabled: true,
                },
                DifficultyEntry {
                    id: "hard".into(),
                    label: "Hard".into(),
                    description: "Tighter economy".into(),
                    config: "hard".into(),
                    reward: "1.5x".into(),
                    image: None,
                    enabled: true,
                },
            ],
            diagnostics: Vec::new(),
            source_path: None,
            used_fallback: false,
        }
    }

    pub fn screen(&self, id: &str) -> Option<&ScreenEntry> {
        self.screens.iter().find(|screen| screen.id == id)
    }

    pub fn map(&self, id: &str) -> Option<&MapEntry> {
        self.maps.iter().find(|map| map.id == id)
    }

    pub fn difficulty(&self, id: &str) -> Option<&DifficultyEntry> {
        self.difficulties
            .iter()
            .find(|difficulty| difficulty.id == id)
    }

    pub fn enabled_maps(&self) -> Vec<&MapEntry> {
        self.maps.iter().filter(|map| map.is_playable()).collect()
    }

    pub fn enabled_difficulties(&self) -> Vec<&DifficultyEntry> {
        self.difficulties
            .iter()
            .filter(|difficulty| difficulty.enabled)
            .collect()
    }

    fn validate_required_session_data(&mut self) {
        for map in &mut self.maps {
            if map.enabled && !map.locked && map.story.trim().is_empty() {
                self.diagnostics.push(format!(
                    "pregame map '{}' missing story/runtime id; disabling",
                    map.id
                ));
                map.enabled = false;
            }
        }
        for difficulty in &mut self.difficulties {
            if difficulty.enabled && difficulty.id.trim().is_empty() {
                self.diagnostics
                    .push("pregame difficulty missing difficulty id; disabling".to_string());
                difficulty.enabled = false;
            }
            if difficulty.enabled && difficulty.config.trim().is_empty() {
                difficulty.config = difficulty.id.clone();
            }
        }
    }

    fn validate_asset_paths(&mut self) {
        let Some(base) = self.source_path.as_ref().and_then(|path| path.parent()) else {
            return;
        };
        for screen in &self.screens {
            log_missing_asset(
                &mut self.diagnostics,
                base,
                screen.background_image.as_deref(),
                &format!("screen {}", screen.id),
            );
            for widget in &screen.widgets {
                log_missing_asset(
                    &mut self.diagnostics,
                    base,
                    widget.image.as_deref(),
                    &format!("widget {}", widget.id),
                );
            }
        }
        for map in &self.maps {
            log_missing_asset(
                &mut self.diagnostics,
                base,
                map.image.as_deref(),
                &format!("map {}", map.id),
            );
        }
        for difficulty in &self.difficulties {
            log_missing_asset(
                &mut self.diagnostics,
                base,
                difficulty.image.as_deref(),
                &format!("difficulty {}", difficulty.id),
            );
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ScreenEntry {
    pub id: String,
    pub title: String,
    pub subtitle: String,
    pub background_image: Option<String>,
    pub widgets: Vec<WidgetEntry>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WidgetEntry {
    pub id: String,
    pub label: String,
    pub description: String,
    pub image: Option<String>,
    pub enabled: bool,
    pub locked: bool,
    pub action: PregameAction,
}

impl WidgetEntry {
    pub fn is_active(&self) -> bool {
        self.enabled && !self.locked
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MapEntry {
    pub id: String,
    pub label: String,
    pub description: String,
    pub story: String,
    pub runtime: String,
    pub image: Option<String>,
    pub enabled: bool,
    pub locked: bool,
    pub reward: String,
}

impl MapEntry {
    pub fn story_id(&self) -> &str {
        if self.runtime.trim().is_empty() {
            &self.story
        } else {
            &self.runtime
        }
    }

    pub fn is_playable(&self) -> bool {
        self.enabled && !self.locked && !self.story_id().trim().is_empty()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DifficultyEntry {
    pub id: String,
    pub label: String,
    pub description: String,
    pub config: String,
    pub reward: String,
    pub image: Option<String>,
    pub enabled: bool,
}

impl DifficultyEntry {
    pub fn config_value(&self) -> &str {
        if self.config.trim().is_empty() {
            &self.id
        } else {
            &self.config
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum PregameAction {
    Navigate {
        target: String,
    },
    Back,
    SelectMap {
        map_id: String,
    },
    SelectDifficulty {
        difficulty_id: String,
    },
    StartSession,
    #[default]
    NoOp,
}

#[derive(Debug, Deserialize)]
struct RawPregameCatalog {
    #[serde(default)]
    screens: Vec<RawScreenEntry>,
    #[serde(default)]
    maps: Vec<RawMapEntry>,
    #[serde(default)]
    difficulties: Vec<RawDifficultyEntry>,
}

#[derive(Debug, Deserialize)]
struct RawScreenEntry {
    id: String,
    title: String,
    #[serde(default)]
    subtitle: Option<String>,
    #[serde(default)]
    background_image: Option<String>,
    #[serde(default)]
    widgets: Vec<RawWidgetEntry>,
}

#[derive(Debug, Deserialize)]
struct RawWidgetEntry {
    id: String,
    label: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    image: Option<String>,
    #[serde(default)]
    enabled: Option<bool>,
    #[serde(default)]
    locked: Option<bool>,
    #[serde(default)]
    action: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct RawMapEntry {
    id: String,
    label: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    story: Option<String>,
    #[serde(default)]
    runtime: Option<String>,
    #[serde(default)]
    image: Option<String>,
    #[serde(default)]
    enabled: Option<bool>,
    #[serde(default)]
    locked: Option<bool>,
    #[serde(default)]
    reward: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawDifficultyEntry {
    id: String,
    label: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    config: Option<String>,
    #[serde(default)]
    reward: Option<String>,
    #[serde(default)]
    image: Option<String>,
    #[serde(default)]
    enabled: Option<bool>,
}

fn parse_action(
    action: serde_json::Value,
    owner_id: &str,
    diagnostics: &mut Vec<String>,
) -> PregameAction {
    let Some(kind) = action.get("kind").and_then(|value| value.as_str()) else {
        diagnostics.push(format!(
            "pregame widget '{}' missing action kind; using NoOp",
            owner_id
        ));
        return PregameAction::NoOp;
    };
    match kind {
        "Navigate" => action
            .get("target")
            .and_then(|value| value.as_str())
            .map(|target| PregameAction::Navigate {
                target: target.to_string(),
            })
            .unwrap_or_else(|| {
                diagnostics.push(format!(
                    "pregame widget '{}' Navigate missing target; using NoOp",
                    owner_id
                ));
                PregameAction::NoOp
            }),
        "Back" => PregameAction::Back,
        "SelectMap" => action
            .get("map_id")
            .or_else(|| action.get("map"))
            .and_then(|value| value.as_str())
            .map(|map_id| PregameAction::SelectMap {
                map_id: map_id.to_string(),
            })
            .unwrap_or_else(|| {
                diagnostics.push(format!(
                    "pregame widget '{}' SelectMap missing map_id; using NoOp",
                    owner_id
                ));
                PregameAction::NoOp
            }),
        "SelectDifficulty" => action
            .get("difficulty_id")
            .or_else(|| action.get("difficulty"))
            .and_then(|value| value.as_str())
            .map(|difficulty_id| PregameAction::SelectDifficulty {
                difficulty_id: difficulty_id.to_string(),
            })
            .unwrap_or_else(|| {
                diagnostics.push(format!(
                    "pregame widget '{}' SelectDifficulty missing difficulty_id; using NoOp",
                    owner_id
                ));
                PregameAction::NoOp
            }),
        "StartSession" => PregameAction::StartSession,
        "NoOp" => PregameAction::NoOp,
        other => {
            diagnostics.push(format!(
                "unknown pregame action '{}' on '{}'; using NoOp",
                other, owner_id
            ));
            PregameAction::NoOp
        }
    }
}

fn candidate_catalog_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Ok(path) = std::env::var("OMFX_PREGAME_CATALOG") {
        paths.push(PathBuf::from(path));
    }
    if let Ok(root) = std::env::var("OMFX_CONTENT_ROOT") {
        paths.push(PathBuf::from(root).join("assets/pregame_ui/catalog.json"));
    }
    if let Ok(root) = std::env::var("OMB_CONTENT_ROOT") {
        paths.push(PathBuf::from(root).join("assets/pregame_ui/catalog.json"));
    }
    paths.push(PathBuf::from(
        "scripts/base_content/assets/pregame_ui/catalog.json",
    ));
    paths.push(PathBuf::from(
        "../scripts/base_content/assets/pregame_ui/catalog.json",
    ));
    paths
}

fn log_missing_asset(
    diagnostics: &mut Vec<String>,
    catalog_dir: &Path,
    asset: Option<&str>,
    owner: &str,
) {
    let Some(asset) = asset else {
        return;
    };
    let trimmed = asset.trim();
    if trimmed.is_empty() {
        return;
    }
    let path = Path::new(trimmed);
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        catalog_dir.join(path)
    };
    if !absolute.exists() {
        diagnostics.push(format!(
            "pregame {} references missing asset {:?}",
            owner, absolute
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_loader_accepts_script_owned_maps_difficulties_and_actions() {
        let json = r#"
        {
          "screens": [
            {
              "id": "main_menu",
              "title": "Mod Lobby",
              "widgets": [
                {"id": "start", "label": "Play", "action": {"kind": "Navigate", "target": "map_select"}}
              ]
            }
          ],
          "maps": [
            {
              "id": "mod_map",
              "label": "Mod Map",
              "story": "TD_MOD",
              "enabled": true,
              "locked": false,
              "image": "assets/pregame_ui/mod_map.png"
            }
          ],
          "difficulties": [
            {
              "id": "nightmare",
              "label": "Nightmare",
              "config": "nightmare",
              "enabled": true
            }
          ]
        }
        "#;

        let catalog = PregameCatalog::from_json_str(json).expect("catalog parses");

        assert_eq!(catalog.screen("main_menu").unwrap().title, "Mod Lobby");
        assert_eq!(catalog.enabled_maps()[0].story, "TD_MOD");
        assert_eq!(catalog.enabled_difficulties()[0].id, "nightmare");
        assert_eq!(
            catalog.screen("main_menu").unwrap().widgets[0].action,
            PregameAction::Navigate {
                target: "map_select".to_string()
            }
        );
    }

    #[test]
    fn malformed_or_unknown_action_is_safe_noop() {
        let json = r#"
        {
          "screens": [
            {
              "id": "main_menu",
              "title": "Main",
              "widgets": [
                {"id": "bad", "label": "Bad", "action": {"kind": "LaunchShell", "target": "x"}}
              ]
            }
          ],
          "maps": [],
          "difficulties": []
        }
        "#;

        let catalog = PregameCatalog::from_json_str(json).expect("catalog parses");

        assert_eq!(
            catalog.screen("main_menu").unwrap().widgets[0].action,
            PregameAction::NoOp
        );
        assert!(catalog
            .diagnostics
            .iter()
            .any(|line| line.contains("unknown pregame action")));
    }

    #[test]
    fn menu_only_runtime_state_has_no_gameplay_resources() {
        let runtime = PregameRuntime::new_for_menu(PregameCatalog::fallback());

        assert!(matches!(runtime.state, PregameState::MainMenu));
        assert!(runtime.selected_map.is_none());
        assert!(runtime.selected_difficulty.is_none());
        assert!(!runtime.has_gameplay_session);
    }
}
