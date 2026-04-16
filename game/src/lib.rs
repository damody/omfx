//! omfx - 2D Tower Defense Game (Fyrox 1.0)
#![allow(warnings)]

use fyrox::graph::prelude::*;
use fyrox::{
    core::{
        algebra::{Vector2, Vector3},
        color::Color,
        pool::Handle,
        reflect::prelude::*,
        visitor::prelude::*,
    },
    event::{ElementState, Event, MouseButton, WindowEvent},
    gui::{
        brush::Brush,
        message::UiMessage,
        text::{Text, TextBuilder, TextMessage},
        widget::WidgetBuilder,
        HorizontalAlignment, UserInterface, VerticalAlignment,
    },
    plugin::{error::GameResult, Plugin, PluginContext, PluginRegistrationContext},
    scene::{
        base::BaseBuilder,
        camera::{CameraBuilder, OrthographicProjection, Projection},
        dim2::rectangle::RectangleBuilder,
        light::{point::PointLightBuilder, BaseLightBuilder},
        node::Node,
        transform::TransformBuilder,
        EnvironmentLightingSource, Scene,
    },
};

pub use fyrox;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WAYPOINTS: &[(f32, f32)] = &[
    (-6.0, 2.0),
    (-2.0, 2.0),
    (-2.0, -1.0),
    (2.0, -1.0),
    (2.0, 2.0),
    (6.0, 2.0),
];

const GRID_COLS: usize = 12;
const GRID_ROWS: usize = 8;
const CELL_SIZE: f32 = 1.0;
const GRID_ORIGIN_X: f32 = -6.0;
const GRID_ORIGIN_Y: f32 = -4.0;

const TOWER_COST: i32 = 20;
const ENEMY_REWARD: i32 = 10;
const INITIAL_GOLD: i32 = 100;
const INITIAL_LIVES: i32 = 20;

const TOWER_RANGE: f32 = 2.5;
const TOWER_DAMAGE: f32 = 25.0;
const TOWER_ATTACK_INTERVAL: f32 = 1.0;

const BULLET_SPEED: f32 = 8.0;

const ENEMY_HP: f32 = 100.0;
const ENEMY_SPEED: f32 = 2.0;

const SPAWN_INTERVAL: f32 = 1.5;
const ENEMIES_PER_WAVE: i32 = 8;

// Z layers: smaller Z = closer to camera (near plane) = renders on top.
// Camera at Z=0, z_near=-0.1. Objects closer to z_near win depth test.
const Z_BULLET: f32 = 0.000;
const Z_HP_BAR: f32 = 0.0005;
const Z_ENEMY: f32 = 0.001;
const Z_TOWER: f32 = 0.002;
const Z_GRID_CELL: f32 = 0.003;
const Z_PATH: f32 = 0.004;
const Z_BACKGROUND: f32 = 0.005;

// ---------------------------------------------------------------------------
// Game Plugin
// ---------------------------------------------------------------------------

#[derive(Default, Visit, Reflect, Debug)]
#[reflect(non_cloneable)]
pub struct Game {
    scene: Handle<Scene>,
    gold: i32,
    lives: i32,
    wave: i32,
    spawn_timer: f32,
    enemies_to_spawn: i32,
    enemies_alive: i32,
    wave_delay: f32,
    game_over: bool,
    camera: Handle<Node>,
    #[visit(skip)]
    #[reflect(hidden)]
    grid: Vec<Vec<bool>>,
    #[visit(skip)]
    #[reflect(hidden)]
    towers: Vec<TowerData>,
    #[visit(skip)]
    #[reflect(hidden)]
    enemies: Vec<EnemyData>,
    #[visit(skip)]
    #[reflect(hidden)]
    bullets: Vec<BulletData>,
    #[visit(skip)]
    #[reflect(hidden)]
    ui_gold_text: Handle<Text>,
    #[visit(skip)]
    #[reflect(hidden)]
    ui_lives_text: Handle<Text>,
    #[visit(skip)]
    #[reflect(hidden)]
    ui_wave_text: Handle<Text>,
    #[visit(skip)]
    #[reflect(hidden)]
    ui_gameover_text: Handle<Text>,
    #[visit(skip)]
    #[reflect(hidden)]
    mouse_world_pos: Vector2<f32>,
    #[visit(skip)]
    #[reflect(hidden)]
    window_size: Vector2<f32>,
}

#[derive(Debug, Clone)]
struct TowerData {
    node: Handle<Node>,
    pos: Vector2<f32>,
    attack_timer: f32,
}

#[derive(Debug, Clone)]
struct EnemyData {
    node: Handle<Node>,
    hp: f32,
    max_hp: f32,
    speed: f32,
    waypoint_index: usize,
    pos: Vector2<f32>,
    hp_bar_bg: Handle<Node>,
    hp_bar_fg: Handle<Node>,
}

#[derive(Debug, Clone)]
struct BulletData {
    node: Handle<Node>,
    target_enemy_idx: usize,
    speed: f32,
    damage: f32,
    pos: Vector2<f32>,
}

impl Plugin for Game {
    fn register(&self, _context: PluginRegistrationContext) -> GameResult {
        Ok(())
    }

    fn init(&mut self, _scene_path: Option<&str>, mut context: PluginContext) -> GameResult {
        self.gold = INITIAL_GOLD;
        self.lives = INITIAL_LIVES;
        self.wave = 0;
        self.spawn_timer = 0.0;
        self.enemies_to_spawn = 0;
        self.enemies_alive = 0;
        self.wave_delay = 2.0;
        self.game_over = false;
        self.window_size = Vector2::new(800.0, 600.0);

        self.grid = vec![vec![true; GRID_COLS]; GRID_ROWS];
        mark_path_cells(&mut self.grid);

        let mut scene = Scene::new();

        // 2D rendering: use ambient color lighting, set clear color
        use fyrox::scene::SceneRenderingOptions;
        scene.rendering_options.set_value_and_mark_modified(SceneRenderingOptions {
            clear_color: Some(Color::from_rgba(30, 80, 30, 255)),
            ambient_lighting_color: Color::WHITE,
            environment_lighting_source: EnvironmentLightingSource::AmbientColor,
            environment_lighting_brightness: 1.0,
            ..Default::default()
        });

        // Orthographic 2D camera at origin (like official 2D example)
        self.camera = CameraBuilder::new(BaseBuilder::new())
            .with_projection(Projection::Orthographic(OrthographicProjection {
                z_near: -0.1,
                z_far: 16.0,
                vertical_size: 5.0,
            }))
            .build(&mut scene.graph)
            .transmute();

        // Point light covering entire map
        PointLightBuilder::new(
            BaseLightBuilder::new(
                BaseBuilder::new().with_local_transform(
                    TransformBuilder::new()
                        .with_local_position(Vector3::new(0.0, 0.0, 0.0))
                        .build(),
                ),
            )
            .with_scatter_enabled(false),
        )
        .with_radius(20.0)
        .build(&mut scene.graph);

        // Background (dark green, furthest layer)
        RectangleBuilder::new(
            BaseBuilder::new().with_local_transform(
                TransformBuilder::new()
                    .with_local_position(Vector3::new(0.0, 0.0, Z_BACKGROUND))
                    .with_local_scale(Vector3::new(14.0, 10.0, f32::EPSILON))
                    .build(),
            ),
        )
        .with_color(Color::from_rgba(30, 80, 30, 255))
        .build(&mut scene.graph);

        // Draw path (grey)
        draw_path(&mut scene);

        // Draw placeable grid cells (light green)
        for row in 0..GRID_ROWS {
            for col in 0..GRID_COLS {
                if self.grid[row][col] {
                    let (cx, cy) = grid_to_world(col, row);
                    RectangleBuilder::new(
                        BaseBuilder::new().with_local_transform(
                            TransformBuilder::new()
                                .with_local_position(Vector3::new(cx, cy, Z_GRID_CELL))
                                .with_local_scale(Vector3::new(
                                    CELL_SIZE * 0.9,
                                    CELL_SIZE * 0.9,
                                    f32::EPSILON,
                                ))
                                .build(),
                        ),
                    )
                    .with_color(Color::from_rgba(60, 140, 60, 80))
                    .build(&mut scene.graph);
                }
            }
        }

        self.scene = context.scenes.add(scene);

        // Create a UI instance
        context
            .user_interfaces
            .add(UserInterface::new(Default::default()));
        let ui = context.user_interfaces.first_mut();

        self.ui_gold_text = TextBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(10.0, 10.0))
                .with_foreground(Brush::Solid(Color::from_rgba(255, 215, 0, 255)).into()),
        )
        .with_text(format!("Gold: {}", self.gold))
        .with_font_size(20.0.into())
        .build(&mut ui.build_ctx());

        self.ui_lives_text = TextBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(150.0, 10.0))
                .with_foreground(Brush::Solid(Color::from_rgba(255, 80, 80, 255)).into()),
        )
        .with_text(format!("Lives: {}", self.lives))
        .with_font_size(20.0.into())
        .build(&mut ui.build_ctx());

        self.ui_wave_text = TextBuilder::new(
            WidgetBuilder::new()
                .with_desired_position(Vector2::new(300.0, 10.0))
                .with_foreground(Brush::Solid(Color::WHITE).into()),
        )
        .with_text("Wave: 0".to_string())
        .with_font_size(20.0.into())
        .build(&mut ui.build_ctx());

        Ok(())
    }

    fn on_deinit(&mut self, _context: PluginContext) -> GameResult {
        Ok(())
    }

    fn update(&mut self, context: &mut PluginContext) -> GameResult {
        if self.game_over {
            return Ok(());
        }

        let dt = context.dt;
        let scene = &mut context.scenes[self.scene];

        // --- Wave spawning ---
        if self.enemies_to_spawn <= 0 && self.enemies_alive <= 0 {
            self.wave_delay -= dt;
            if self.wave_delay <= 0.0 {
                self.wave += 1;
                self.enemies_to_spawn = ENEMIES_PER_WAVE + (self.wave - 1) * 2;
                self.spawn_timer = 0.0;
                self.wave_delay = 3.0;
            }
        }

        if self.enemies_to_spawn > 0 {
            self.spawn_timer -= dt;
            if self.spawn_timer <= 0.0 {
                self.spawn_timer = SPAWN_INTERVAL;
                self.enemies_to_spawn -= 1;
                self.enemies_alive += 1;

                let start = WAYPOINTS[0];
                let node = RectangleBuilder::new(
                    BaseBuilder::new().with_local_transform(
                        TransformBuilder::new()
                            .with_local_position(Vector3::new(start.0, start.1, Z_ENEMY))
                            .with_local_scale(Vector3::new(0.3, 0.3, f32::EPSILON))
                            .build(),
                    ),
                )
                .with_color(Color::from_rgba(220, 40, 40, 255))
                .build(&mut scene.graph)
                .transmute();

                let bar_y = start.1 + 0.25;
                let hp_bar_bg = RectangleBuilder::new(
                    BaseBuilder::new().with_local_transform(
                        TransformBuilder::new()
                            .with_local_position(Vector3::new(start.0, bar_y, Z_HP_BAR))
                            .with_local_scale(Vector3::new(0.4, 0.06, f32::EPSILON))
                            .build(),
                    ),
                )
                .with_color(Color::from_rgba(40, 40, 40, 200))
                .build(&mut scene.graph)
                .transmute();

                let hp_bar_fg = RectangleBuilder::new(
                    BaseBuilder::new().with_local_transform(
                        TransformBuilder::new()
                            .with_local_position(Vector3::new(start.0, bar_y, Z_HP_BAR - 0.0001))
                            .with_local_scale(Vector3::new(0.4, 0.06, f32::EPSILON))
                            .build(),
                    ),
                )
                .with_color(Color::from_rgba(0, 220, 0, 255))
                .build(&mut scene.graph)
                .transmute();

                let enemy_max_hp = ENEMY_HP + (self.wave as f32 - 1.0) * 20.0;
                self.enemies.push(EnemyData {
                    node,
                    hp: enemy_max_hp,
                    max_hp: enemy_max_hp,
                    speed: ENEMY_SPEED,
                    waypoint_index: 1,
                    pos: Vector2::new(start.0, start.1),
                    hp_bar_bg,
                    hp_bar_fg,
                });
            }
        }

        // --- Enemy movement ---
        let mut enemies_to_remove = Vec::new();
        for (i, enemy) in self.enemies.iter_mut().enumerate() {
            if enemy.waypoint_index >= WAYPOINTS.len() {
                enemies_to_remove.push(i);
                self.lives -= 1;
                self.enemies_alive -= 1;
                continue;
            }

            let target = WAYPOINTS[enemy.waypoint_index];
            let target_pos = Vector2::new(target.0, target.1);
            let dir = target_pos - enemy.pos;
            let dist = dir.norm();

            if dist < 0.05 {
                enemy.waypoint_index += 1;
            } else {
                let movement = dir.normalize() * enemy.speed * dt;
                enemy.pos += movement;
                scene.graph[enemy.node]
                    .local_transform_mut()
                    .set_position(Vector3::new(enemy.pos.x, enemy.pos.y, Z_ENEMY));
            }

            // Update health bar position & width
            let bar_y = enemy.pos.y + 0.25;
            let hp_ratio = (enemy.hp / enemy.max_hp).clamp(0.0, 1.0);
            let bar_width = 0.4;

            scene.graph[enemy.hp_bar_bg]
                .local_transform_mut()
                .set_position(Vector3::new(enemy.pos.x, bar_y, Z_HP_BAR));

            let fg_width = bar_width * hp_ratio;
            let fg_offset = (bar_width - fg_width) * 0.5;
            scene.graph[enemy.hp_bar_fg]
                .local_transform_mut()
                .set_position(Vector3::new(enemy.pos.x - fg_offset, bar_y, Z_HP_BAR - 0.0001))
                .set_scale(Vector3::new(fg_width, 0.06, f32::EPSILON));

            // Color: green → yellow → red
            let r = ((1.0 - hp_ratio) * 2.0).min(1.0);
            let g = (hp_ratio * 2.0).min(1.0);
            scene.graph[enemy.hp_bar_fg]
                .as_rectangle_mut()
                .set_color(Color::from_rgba((r * 255.0) as u8, (g * 255.0) as u8, 0, 255));
        }

        for &i in enemies_to_remove.iter().rev() {
            let enemy = self.enemies.remove(i);
            scene.graph.remove_node(enemy.node);
            scene.graph.remove_node(enemy.hp_bar_bg);
            scene.graph.remove_node(enemy.hp_bar_fg);
        }

        // --- Tower attacks ---
        for tower in self.towers.iter_mut() {
            tower.attack_timer -= dt;
            if tower.attack_timer <= 0.0 {
                let mut closest_idx = None;
                let mut closest_dist = f32::MAX;

                for (i, enemy) in self.enemies.iter().enumerate() {
                    let dist = (enemy.pos - tower.pos).norm();
                    if dist < TOWER_RANGE && dist < closest_dist {
                        closest_dist = dist;
                        closest_idx = Some(i);
                    }
                }

                if let Some(target_idx) = closest_idx {
                    tower.attack_timer = TOWER_ATTACK_INTERVAL;

                    let bullet_node = RectangleBuilder::new(
                        BaseBuilder::new().with_local_transform(
                            TransformBuilder::new()
                                .with_local_position(Vector3::new(
                                    tower.pos.x,
                                    tower.pos.y,
                                    Z_BULLET,
                                ))
                                .with_local_scale(Vector3::new(0.1, 0.1, f32::EPSILON))
                                .build(),
                        ),
                    )
                    .with_color(Color::from_rgba(255, 230, 50, 255))
                    .build(&mut scene.graph)
                    .transmute();

                    self.bullets.push(BulletData {
                        node: bullet_node,
                        target_enemy_idx: target_idx,
                        speed: BULLET_SPEED,
                        damage: TOWER_DAMAGE,
                        pos: tower.pos,
                    });
                }
            }
        }

        // --- Bullet movement ---
        let mut bullets_to_remove = Vec::new();
        let mut enemy_damage: Vec<(usize, f32)> = Vec::new();

        for (bi, bullet) in self.bullets.iter_mut().enumerate() {
            if bullet.target_enemy_idx >= self.enemies.len() {
                bullets_to_remove.push(bi);
                continue;
            }

            let target_pos = self.enemies[bullet.target_enemy_idx].pos;
            let dir = target_pos - bullet.pos;
            let dist = dir.norm();

            if dist < 0.15 {
                enemy_damage.push((bullet.target_enemy_idx, bullet.damage));
                bullets_to_remove.push(bi);
            } else {
                let movement = dir.normalize() * bullet.speed * dt;
                bullet.pos += movement;
                scene.graph[bullet.node]
                    .local_transform_mut()
                    .set_position(Vector3::new(bullet.pos.x, bullet.pos.y, Z_BULLET));
            }
        }

        for &i in bullets_to_remove.iter().rev() {
            let bullet = self.bullets.remove(i);
            scene.graph.remove_node(bullet.node);
        }

        // Apply damage
        let mut dead_enemies = Vec::new();
        for (idx, dmg) in enemy_damage {
            if idx < self.enemies.len() {
                self.enemies[idx].hp -= dmg;
                if self.enemies[idx].hp <= 0.0 {
                    dead_enemies.push(idx);
                }
            }
        }

        dead_enemies.sort_unstable();
        dead_enemies.dedup();
        for &i in dead_enemies.iter().rev() {
            let enemy = self.enemies.remove(i);
            scene.graph.remove_node(enemy.node);
            scene.graph.remove_node(enemy.hp_bar_bg);
            scene.graph.remove_node(enemy.hp_bar_fg);
            self.gold += ENEMY_REWARD;
            self.enemies_alive -= 1;

            for bullet in self.bullets.iter_mut() {
                if bullet.target_enemy_idx == i {
                    bullet.target_enemy_idx = usize::MAX;
                } else if bullet.target_enemy_idx > i {
                    bullet.target_enemy_idx -= 1;
                }
            }
        }

        // --- Check game over ---
        if self.lives <= 0 {
            self.game_over = true;
            let ui = context.user_interfaces.first_mut();
            self.ui_gameover_text = TextBuilder::new(
                WidgetBuilder::new()
                    .with_desired_position(Vector2::new(0.0, 0.0))
                    .with_width(800.0)
                    .with_height(600.0)
                    .with_foreground(Brush::Solid(Color::RED).into()),
            )
            .with_text("GAME OVER".to_string())
            .with_font_size(48.0.into())
            .with_horizontal_text_alignment(HorizontalAlignment::Center)
            .with_vertical_text_alignment(VerticalAlignment::Center)
            .build(&mut ui.build_ctx());
        }

        // --- Update UI text ---
        let ui = context.user_interfaces.first_mut();
        ui.send(self.ui_gold_text, TextMessage::Text(format!("Gold: {}", self.gold)));
        ui.send(self.ui_lives_text, TextMessage::Text(format!("Lives: {}", self.lives)));
        ui.send(self.ui_wave_text, TextMessage::Text(format!("Wave: {}", self.wave)));

        Ok(())
    }

    fn on_os_event(&mut self, event: &Event<()>, mut context: PluginContext) -> GameResult {
        if self.game_over {
            return Ok(());
        }

        match event {
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                self.window_size = Vector2::new(size.width as f32, size.height as f32);
            }
            Event::WindowEvent {
                event: WindowEvent::CursorMoved { position, .. },
                ..
            } => {
                self.mouse_world_pos = screen_to_world_approx(
                    position.x as f32,
                    position.y as f32,
                    self.window_size.x,
                    self.window_size.y,
                    10.0,
                );
            }
            Event::WindowEvent {
                event:
                    WindowEvent::MouseInput {
                        button: MouseButton::Left,
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                if self.gold >= TOWER_COST {
                    let world_pos = self.mouse_world_pos;
                    if let Some((col, row)) = world_to_grid(world_pos.x, world_pos.y) {
                        if row < GRID_ROWS && col < GRID_COLS && self.grid[row][col] {
                            self.grid[row][col] = false;
                            self.gold -= TOWER_COST;

                            let (cx, cy) = grid_to_world(col, row);
                            let scene = &mut context.scenes[self.scene];
                            let node: Handle<Node> = RectangleBuilder::new(
                                BaseBuilder::new().with_local_transform(
                                    TransformBuilder::new()
                                        .with_local_position(Vector3::new(cx, cy, Z_TOWER))
                                        .with_local_scale(Vector3::new(0.4, 0.4, f32::EPSILON))
                                        .build(),
                                ),
                            )
                            .with_color(Color::from_rgba(50, 100, 220, 255))
                            .build(&mut scene.graph)
                            .transmute();

                            self.towers.push(TowerData {
                                node,
                                pos: Vector2::new(cx, cy),
                                attack_timer: 0.0,
                            });
                        }
                    }
                }
            }
            _ => {}
        }

        Ok(())
    }

    fn on_ui_message(
        &mut self,
        _context: &mut PluginContext,
        _message: &UiMessage,
        _ui_handle: Handle<UserInterface>,
    ) -> GameResult {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn grid_to_world(col: usize, row: usize) -> (f32, f32) {
    let x = GRID_ORIGIN_X + col as f32 * CELL_SIZE + CELL_SIZE * 0.5;
    let y = GRID_ORIGIN_Y + row as f32 * CELL_SIZE + CELL_SIZE * 0.5;
    (x, y)
}

fn world_to_grid(wx: f32, wy: f32) -> Option<(usize, usize)> {
    let col = ((wx - GRID_ORIGIN_X) / CELL_SIZE).floor() as i32;
    let row = ((wy - GRID_ORIGIN_Y) / CELL_SIZE).floor() as i32;
    if col >= 0 && col < GRID_COLS as i32 && row >= 0 && row < GRID_ROWS as i32 {
        Some((col as usize, row as usize))
    } else {
        None
    }
}

fn screen_to_world_approx(
    screen_x: f32,
    screen_y: f32,
    window_w: f32,
    window_h: f32,
    world_height: f32,
) -> Vector2<f32> {
    let aspect = window_w / window_h;
    let world_width = world_height * aspect;
    let wx = -(screen_x / window_w - 0.5) * world_width;
    let wy = -(screen_y / window_h - 0.5) * world_height;
    Vector2::new(wx, wy)
}

fn mark_path_cells(grid: &mut Vec<Vec<bool>>) {
    for i in 0..WAYPOINTS.len() - 1 {
        let (x0, y0) = WAYPOINTS[i];
        let (x1, y1) = WAYPOINTS[i + 1];

        let steps = ((x1 - x0).abs().max((y1 - y0).abs()) / (CELL_SIZE * 0.5)) as usize + 1;
        for s in 0..=steps {
            let t = s as f32 / steps as f32;
            let px = x0 + (x1 - x0) * t;
            let py = y0 + (y1 - y0) * t;

            if let Some((col, row)) = world_to_grid(px, py) {
                if row < GRID_ROWS && col < GRID_COLS {
                    grid[row][col] = false;
                    if col > 0 {
                        grid[row][col - 1] = false;
                    }
                    if col + 1 < GRID_COLS {
                        grid[row][col + 1] = false;
                    }
                }
            }
        }
    }
}

fn draw_path(scene: &mut Scene) {
    for i in 0..WAYPOINTS.len() - 1 {
        let (x0, y0) = WAYPOINTS[i];
        let (x1, y1) = WAYPOINTS[i + 1];

        let steps = ((x1 - x0).abs().max((y1 - y0).abs()) / 0.3) as usize + 1;
        for s in 0..=steps {
            let t = s as f32 / steps as f32;
            let px = x0 + (x1 - x0) * t;
            let py = y0 + (y1 - y0) * t;

            RectangleBuilder::new(
                BaseBuilder::new().with_local_transform(
                    TransformBuilder::new()
                        .with_local_position(Vector3::new(px, py, Z_PATH))
                        .with_local_scale(Vector3::new(0.8, 0.8, f32::EPSILON))
                        .build(),
                ),
            )
            .with_color(Color::from_rgba(120, 120, 120, 255))
            .build(&mut scene.graph);
        }
    }
}
