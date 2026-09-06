//! Presentation-only bridge for the disclosed replica world. Remembered
//! ghosts are deliberately stored outside target, collision, simulation and
//! hash state.

use std::collections::{BTreeMap, BTreeSet};

use omoba_core::runtime::{FilteredRenderEntity, FilteredRenderSnapshot, RenderMemoryDirective};

fn presentation_trace_enabled() -> bool {
    std::env::var("OMOBA_PRESENTATION_TRACE")
        .ok()
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct OwnedHeroPresentation {
    pub render_id: u64,
    pub name: String,
    pub title: String,
    pub level: i32,
    pub experience: i32,
    pub experience_to_next: i32,
    pub skill_points: i32,
    pub strength: i32,
    pub agility: i32,
    pub intelligence: i32,
    pub primary_attribute: String,
    pub hp: f32,
    pub max_hp: f32,
    pub armor: f32,
    pub magic_resist: f32,
    pub move_speed: f32,
    pub attack_damage: f32,
    pub attack_interval: f32,
    pub attack_range: f32,
    pub bullet_speed: f32,
    pub abilities: Vec<String>,
    pub ability_levels: std::collections::HashMap<String, i32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct RememberedAssociationKey {
    pub replica_id: u64,
    pub disclosure_epoch: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RememberedPresentation {
    pub key: RememberedAssociationKey,
    pub presentation_kind: u32,
    pub sanitized_payload: Vec<u8>,
    pub expires_at_tick: Option<u64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RememberPresentationRule {
    pub ttl_ticks: Option<u64>,
}

#[derive(Clone, Debug, Default)]
pub struct RememberPresentationRegistry {
    rules: BTreeMap<u32, RememberPresentationRule>,
}

impl RememberPresentationRegistry {
    pub fn register(&mut self, policy: u32, rule: RememberPresentationRule) {
        self.rules.insert(policy, rule);
    }
    pub fn lookup(&self, policy: u32) -> Option<RememberPresentationRule> {
        self.rules.get(&policy).copied()
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FilteredSceneAction {
    UpsertDeterministic(FilteredRenderEntity),
    RemoveDeterministicNode(u64),
    RetireDeterministicIdentity(u64),
    InsertRemembered(RememberedPresentation),
    RemoveRemembered(RememberedAssociationKey),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RenderLifecycleAction {
    Hide {
        replica_id: u64,
        disclosure_epoch: u64,
        remember_policy: u32,
        sanitized_presentation: Vec<u8>,
    },
    Forget {
        replica_id: u64,
        disclosure_epoch: u64,
    },
    ResetView,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RenderLifecycleBatch {
    pub sequence: u64,
    pub team_id: u32,
    pub authoritative_tick: u64,
    pub replica_tick: u64,
    pub view_epoch: u64,
    pub events: Vec<RenderLifecycleAction>,
}

#[derive(Debug)]
pub struct FilteredRenderBridge {
    deterministic: BTreeMap<u64, FilteredRenderEntity>,
    remembered: BTreeMap<RememberedAssociationKey, RememberedPresentation>,
    closed_disclosure_epochs: BTreeMap<u64, u64>,
    retired: BTreeSet<u64>,
    registry: RememberPresentationRegistry,
    view_epoch: Option<u64>,
    last_lifecycle_sequence: u64,
    last_trace_stale_ids: BTreeSet<u64>,
    last_trace_heroes: Vec<String>,
}

impl Default for FilteredRenderBridge {
    fn default() -> Self {
        let mut registry = RememberPresentationRegistry::default();
        registry.register(
            1,
            RememberPresentationRule {
                ttl_ticks: Some(120 * 10),
            },
        );
        Self {
            deterministic: BTreeMap::new(),
            remembered: BTreeMap::new(),
            closed_disclosure_epochs: BTreeMap::new(),
            retired: BTreeSet::new(),
            registry,
            view_epoch: None,
            last_lifecycle_sequence: 0,
            last_trace_stale_ids: BTreeSet::new(),
            last_trace_heroes: Vec::new(),
        }
    }
}

impl FilteredRenderBridge {
    pub fn deterministic_count(&self) -> usize {
        self.deterministic.len()
    }

    pub fn remembered_count(&self) -> usize {
        self.remembered.len()
    }

    pub fn apply_lifecycle(&mut self, batch: &RenderLifecycleBatch) -> Vec<FilteredSceneAction> {
        let has_reset = batch
            .events
            .iter()
            .any(|event| matches!(event, RenderLifecycleAction::ResetView));
        if batch.sequence <= self.last_lifecycle_sequence && !has_reset {
            return Vec::new();
        }
        match self.view_epoch {
            Some(current) if batch.view_epoch < current => return Vec::new(),
            Some(current) if batch.view_epoch > current && !has_reset => return Vec::new(),
            None if !has_reset => return Vec::new(),
            _ => {}
        }
        self.last_lifecycle_sequence = batch.sequence;
        let mut actions = Vec::new();
        self.expire(batch.replica_tick, &mut actions);
        for event in &batch.events {
            match event {
                RenderLifecycleAction::ResetView => {
                    self.reset_view(batch.view_epoch, &mut actions);
                }
                RenderLifecycleAction::Hide {
                    replica_id,
                    disclosure_epoch,
                    remember_policy,
                    sanitized_presentation,
                } => self.apply_directive(
                    batch.replica_tick,
                    &RenderMemoryDirective::Hide {
                        replica_id: *replica_id,
                        disclosure_epoch: *disclosure_epoch,
                        remember_policy: *remember_policy,
                        sanitized_presentation: sanitized_presentation.clone(),
                    },
                    &mut actions,
                ),
                RenderLifecycleAction::Forget {
                    replica_id,
                    disclosure_epoch,
                } => self.apply_directive(
                    batch.replica_tick,
                    &RenderMemoryDirective::Forget {
                        replica_id: *replica_id,
                        disclosure_epoch: *disclosure_epoch,
                    },
                    &mut actions,
                ),
            }
        }
        actions
    }

    pub fn deterministic_demo_states(
        &self,
    ) -> impl Iterator<Item = (u64, omoba_core::runtime::DemoRenderState)> + '_ {
        self.deterministic.iter().filter_map(|(id, entity)| {
            entity
                .components
                .get(&omoba_core::runtime::DEMO_RENDER_COMPONENT_SCHEMA_ID)
                .and_then(|bytes| omoba_core::runtime::decode_demo_render_state(bytes))
                .map(|state| (*id, state))
        })
    }

    pub fn owned_hero_presentation(&self, player_id: u32) -> Option<OwnedHeroPresentation> {
        use omoba_core::runtime::{
            CProperty, Hero, TAttack, DEMO_RENDER_COMPONENT_SCHEMA_ID,
            DISCLOSED_ATTACK_COMPONENT_SCHEMA_ID, DISCLOSED_HERO_COMPONENT_SCHEMA_ID,
            DISCLOSED_PROPERTY_COMPONENT_SCHEMA_ID,
        };
        self.deterministic.iter().find_map(|(render_id, entity)| {
            let render = entity
                .components
                .get(&DEMO_RENDER_COMPONENT_SCHEMA_ID)
                .and_then(|bytes| omoba_core::runtime::decode_demo_render_state(bytes))?;
            if render.kind != 1 || render.owner_player_id != player_id {
                return None;
            }
            let hero: Hero =
                serde_json::from_slice(entity.components.get(&DISCLOSED_HERO_COMPONENT_SCHEMA_ID)?)
                    .ok()?;
            let property_bytes = entity
                .components
                .get(&DISCLOSED_PROPERTY_COMPONENT_SCHEMA_ID)?;
            if property_bytes.len() != 40 {
                return None;
            }
            let raw = |offset| {
                omoba_sim::Fixed64::from_raw(i64::from_be_bytes(
                    property_bytes[offset..offset + 8].try_into().unwrap(),
                ))
                .to_f32_for_render()
            };
            let property = CProperty {
                hp: omoba_sim::Fixed64::from_raw(i64::from_be_bytes(
                    property_bytes[0..8].try_into().unwrap(),
                )),
                mhp: omoba_sim::Fixed64::from_raw(i64::from_be_bytes(
                    property_bytes[8..16].try_into().unwrap(),
                )),
                msd: omoba_sim::Fixed64::from_raw(i64::from_be_bytes(
                    property_bytes[16..24].try_into().unwrap(),
                )),
                def_physic: omoba_sim::Fixed64::from_raw(i64::from_be_bytes(
                    property_bytes[24..32].try_into().unwrap(),
                )),
                def_magic: omoba_sim::Fixed64::from_raw(i64::from_be_bytes(
                    property_bytes[32..40].try_into().unwrap(),
                )),
            };
            let attack: Option<TAttack> = entity
                .components
                .get(&DISCLOSED_ATTACK_COMPONENT_SCHEMA_ID)
                .and_then(|bytes| serde_json::from_slice(bytes).ok());
            Some(OwnedHeroPresentation {
                render_id: *render_id,
                name: hero.name,
                title: hero.title,
                level: hero.level,
                experience: hero.experience,
                experience_to_next: hero.experience_to_next,
                skill_points: hero.skill_points,
                strength: hero.strength,
                agility: hero.agility,
                intelligence: hero.intelligence,
                primary_attribute: format!("{:?}", hero.primary_attribute).to_lowercase(),
                hp: property.hp.to_f32_for_render(),
                max_hp: property.mhp.to_f32_for_render(),
                armor: property.def_physic.to_f32_for_render(),
                magic_resist: property.def_magic.to_f32_for_render(),
                move_speed: raw(16),
                attack_damage: attack
                    .as_ref()
                    .map_or(0.0, |value| value.atk_physic.v.to_f32_for_render()),
                attack_interval: attack
                    .as_ref()
                    .map_or(0.0, |value| value.asd.v.to_f32_for_render()),
                attack_range: attack
                    .as_ref()
                    .map_or(0.0, |value| value.range.v.to_f32_for_render()),
                bullet_speed: attack
                    .as_ref()
                    .map_or(0.0, |value| value.bullet_speed.to_f32_for_render()),
                abilities: hero.abilities,
                ability_levels: hero.ability_levels,
            })
        })
    }

    pub fn remembered_demo_states(
        &self,
    ) -> impl Iterator<
        Item = (
            RememberedAssociationKey,
            omoba_core::runtime::DemoRenderState,
        ),
    > + '_ {
        self.remembered.iter().filter_map(|(key, presentation)| {
            omoba_core::runtime::decode_demo_render_state(&presentation.sanitized_payload)
                .map(|state| (*key, state))
        })
    }

    pub fn apply(&mut self, snapshot: &FilteredRenderSnapshot) -> Vec<FilteredSceneAction> {
        let mut actions = Vec::new();
        let incoming_ids = snapshot
            .entities
            .iter()
            .map(|entity| entity.replica_id)
            .collect::<BTreeSet<_>>();
        self.expire(snapshot.replica_tick, &mut actions);
        for directive in &snapshot.memory_directives {
            self.apply_directive(snapshot.replica_tick, directive, &mut actions);
        }
        for entity in &snapshot.entities {
            let key = RememberedAssociationKey {
                replica_id: entity.replica_id,
                disclosure_epoch: entity.disclosure_epoch,
            };
            let disclosure_is_closed = self
                .closed_disclosure_epochs
                .get(&entity.replica_id)
                .is_some_and(|closed_epoch| entity.disclosure_epoch <= *closed_epoch);
            if self.retired.contains(&entity.replica_id) || disclosure_is_closed {
                if presentation_trace_enabled() {
                    log::warn!(
                        "presentation_trace stage=render_bridge_stale_snapshot_ignored replica_tick={} replica_id={} disclosure_epoch={}",
                        snapshot.replica_tick,
                        entity.replica_id,
                        entity.disclosure_epoch,
                    );
                }
                continue;
            }
            let remembered_keys = self
                .remembered
                .keys()
                .filter(|remembered_key| remembered_key.replica_id == entity.replica_id)
                .copied()
                .collect::<Vec<_>>();
            for remembered_key in remembered_keys {
                self.remembered.remove(&remembered_key);
                actions.push(FilteredSceneAction::RemoveRemembered(remembered_key));
            }
            self.deterministic.insert(entity.replica_id, entity.clone());
            actions.push(FilteredSceneAction::UpsertDeterministic(entity.clone()));
        }
        if presentation_trace_enabled() {
            let stale_ids = self
                .deterministic
                .keys()
                .filter(|replica_id| !incoming_ids.contains(replica_id))
                .copied()
                .collect::<BTreeSet<_>>();
            let heroes = self
                .deterministic
                .values()
                .filter_map(|entity| {
                    let payload = entity
                        .components
                        .get(&omoba_core::runtime::DEMO_RENDER_COMPONENT_SCHEMA_ID)?;
                    let state = omoba_core::runtime::decode_demo_render_state(payload)?;
                    (state.kind == 1).then(|| {
                        format!(
                            "{}@{}:player{}:team{}:pos({}, {})",
                            entity.replica_id,
                            entity.disclosure_epoch,
                            state.owner_player_id,
                            state.team_id,
                            state.x_raw,
                            state.y_raw,
                        )
                    })
                })
                .collect::<Vec<_>>();
            let hero_identities = self
                .deterministic
                .values()
                .filter_map(|entity| {
                    let payload = entity
                        .components
                        .get(&omoba_core::runtime::DEMO_RENDER_COMPONENT_SCHEMA_ID)?;
                    let state = omoba_core::runtime::decode_demo_render_state(payload)?;
                    (state.kind == 1).then(|| {
                        format!(
                            "{}@{}:player{}:team{}",
                            entity.replica_id,
                            entity.disclosure_epoch,
                            state.owner_player_id,
                            state.team_id,
                        )
                    })
                })
                .collect::<Vec<_>>();
            if stale_ids != self.last_trace_stale_ids || hero_identities != self.last_trace_heroes {
                log::warn!(
                    "presentation_trace stage=render_bridge_reconcile_probe replica_tick={} stale_deterministic_ids={:?} heroes={:?}",
                    snapshot.replica_tick,
                    stale_ids,
                    heroes,
                );
                self.last_trace_stale_ids = stale_ids;
                self.last_trace_heroes = hero_identities;
            }
        }
        actions
    }

    fn apply_directive(
        &mut self,
        replica_tick: u64,
        directive: &RenderMemoryDirective,
        actions: &mut Vec<FilteredSceneAction>,
    ) {
        match directive {
            RenderMemoryDirective::Hide {
                replica_id,
                disclosure_epoch,
                remember_policy,
                sanitized_presentation,
            } => {
                if self.deterministic.remove(replica_id).is_some() {
                    actions.push(FilteredSceneAction::RemoveDeterministicNode(*replica_id));
                }
                self.closed_disclosure_epochs
                    .entry(*replica_id)
                    .and_modify(|closed_epoch| {
                        *closed_epoch = (*closed_epoch).max(*disclosure_epoch);
                    })
                    .or_insert(*disclosure_epoch);
                if let Some(rule) = self.registry.lookup(*remember_policy) {
                    let presentation = RememberedPresentation {
                        key: RememberedAssociationKey {
                            replica_id: *replica_id,
                            disclosure_epoch: *disclosure_epoch,
                        },
                        presentation_kind: *remember_policy,
                        sanitized_payload: sanitized_presentation.clone(),
                        expires_at_tick: rule.ttl_ticks.map(|ttl| replica_tick.saturating_add(ttl)),
                    };
                    if self.remembered.get(&presentation.key) != Some(&presentation) {
                        self.remembered
                            .insert(presentation.key, presentation.clone());
                        actions.push(FilteredSceneAction::InsertRemembered(presentation));
                    }
                }
            }
            RenderMemoryDirective::Forget {
                replica_id,
                disclosure_epoch,
            } => {
                if self.deterministic.remove(replica_id).is_some() {
                    actions.push(FilteredSceneAction::RemoveDeterministicNode(*replica_id));
                }
                let key = RememberedAssociationKey {
                    replica_id: *replica_id,
                    disclosure_epoch: *disclosure_epoch,
                };
                if self.remembered.remove(&key).is_some() {
                    actions.push(FilteredSceneAction::RemoveRemembered(key));
                }
                if self.retired.insert(*replica_id) {
                    actions.push(FilteredSceneAction::RetireDeterministicIdentity(
                        *replica_id,
                    ));
                }
            }
        }
    }

    fn reset_view(&mut self, view_epoch: u64, actions: &mut Vec<FilteredSceneAction>) {
        for replica_id in self.deterministic.keys().copied().collect::<Vec<_>>() {
            actions.push(FilteredSceneAction::RemoveDeterministicNode(replica_id));
        }
        for key in self.remembered.keys().copied().collect::<Vec<_>>() {
            actions.push(FilteredSceneAction::RemoveRemembered(key));
        }
        self.deterministic.clear();
        self.remembered.clear();
        self.closed_disclosure_epochs.clear();
        self.retired.clear();
        self.view_epoch = Some(view_epoch);
    }

    fn expire(&mut self, tick: u64, actions: &mut Vec<FilteredSceneAction>) {
        let expired: Vec<_> = self
            .remembered
            .iter()
            .filter(|(_, value)| value.expires_at_tick.is_some_and(|expiry| expiry <= tick))
            .map(|(key, _)| *key)
            .collect();
        for key in expired {
            self.remembered.remove(&key);
            actions.push(FilteredSceneAction::RemoveRemembered(key));
        }
    }

    pub fn remove_remembered(&mut self, key: RememberedAssociationKey) -> bool {
        self.remembered.remove(&key).is_some()
    }

    pub fn deterministic_entity(&self, replica_id: u64) -> Option<&FilteredRenderEntity> {
        self.deterministic.get(&replica_id)
    }

    // No target/collision/simulation/hash accessor exists for remembered data.
    pub fn remembered_presentation(
        &self,
        key: RememberedAssociationKey,
    ) -> Option<&RememberedPresentation> {
        self.remembered.get(&key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use omoba_core::runtime::{
        encode_disclosed_property, CProperty, DemoRenderState, Hero,
        DEMO_RENDER_COMPONENT_SCHEMA_ID, DISCLOSED_HERO_COMPONENT_SCHEMA_ID,
        DISCLOSED_PROPERTY_COMPONENT_SCHEMA_ID,
    };
    use omoba_sim::Fixed64;

    fn reset(sequence: u64, view_epoch: u64) -> RenderLifecycleBatch {
        RenderLifecycleBatch {
            sequence,
            team_id: 1,
            authoritative_tick: sequence,
            replica_tick: sequence,
            view_epoch,
            events: vec![RenderLifecycleAction::ResetView],
        }
    }

    fn entity(replica_id: u64, disclosure_epoch: u64) -> FilteredRenderEntity {
        FilteredRenderEntity {
            replica_id,
            disclosure_epoch,
            entity_kind: 1,
            components: BTreeMap::new(),
        }
    }

    fn snapshot(replica_tick: u64, entities: Vec<FilteredRenderEntity>) -> FilteredRenderSnapshot {
        FilteredRenderSnapshot {
            team_id: 1,
            replica_tick,
            entities,
            public_events: Vec::new(),
            external_effects: Vec::new(),
            memory_directives: Vec::new(),
        }
    }

    #[test]
    fn duplicate_forget_is_idempotent() {
        let mut bridge = FilteredRenderBridge::default();
        bridge.apply_lifecycle(&reset(1, 7));
        bridge.apply(&snapshot(2, vec![entity(42, 3)]));
        let forget = RenderLifecycleAction::Forget {
            replica_id: 42,
            disclosure_epoch: 3,
        };
        let first = bridge.apply_lifecycle(&RenderLifecycleBatch {
            sequence: 2,
            team_id: 1,
            authoritative_tick: 3,
            replica_tick: 3,
            view_epoch: 7,
            events: vec![forget.clone()],
        });
        let second = bridge.apply_lifecycle(&RenderLifecycleBatch {
            sequence: 3,
            team_id: 1,
            authoritative_tick: 4,
            replica_tick: 4,
            view_epoch: 7,
            events: vec![forget],
        });
        assert!(first
            .iter()
            .any(|action| matches!(action, FilteredSceneAction::RemoveDeterministicNode(42))));
        assert!(second.is_empty());
        assert_eq!(bridge.deterministic_count(), 0);
    }

    #[test]
    fn hide_blocks_late_snapshot_and_higher_epoch_reveals_once() {
        let mut bridge = FilteredRenderBridge::default();
        bridge.apply_lifecycle(&reset(1, 7));
        bridge.apply(&snapshot(2, vec![entity(42, 3)]));
        bridge.apply_lifecycle(&RenderLifecycleBatch {
            sequence: 2,
            team_id: 1,
            authoritative_tick: 3,
            replica_tick: 3,
            view_epoch: 7,
            events: vec![RenderLifecycleAction::Hide {
                replica_id: 42,
                disclosure_epoch: 3,
                remember_policy: 1,
                sanitized_presentation: vec![1, 2, 3],
            }],
        });
        assert_eq!(bridge.deterministic_count(), 0);
        assert_eq!(bridge.remembered_count(), 1);
        let stale_actions = bridge.apply(&snapshot(2, vec![entity(42, 3)]));
        assert!(stale_actions.iter().all(|action| !matches!(
            action,
            FilteredSceneAction::UpsertDeterministic(entity) if entity.replica_id == 42
        )));
        assert_eq!(bridge.deterministic_count(), 0);
        assert_eq!(bridge.remembered_count(), 1);
        bridge.apply(&snapshot(4, vec![entity(42, 4)]));
        assert_eq!(bridge.deterministic_count(), 1);
        assert_eq!(bridge.remembered_count(), 0);
        assert_eq!(bridge.deterministic_entity(42).unwrap().disclosure_epoch, 4);

        bridge.apply(&snapshot(2, vec![entity(42, 3)]));
        assert_eq!(bridge.deterministic_count(), 1);
        assert_eq!(bridge.deterministic_entity(42).unwrap().disclosure_epoch, 4);
    }

    #[test]
    fn forget_blocks_late_snapshot_from_resurrecting_retired_identity() {
        let mut bridge = FilteredRenderBridge::default();
        bridge.apply_lifecycle(&reset(1, 7));
        bridge.apply(&snapshot(2, vec![entity(105, 1)]));
        bridge.apply_lifecycle(&RenderLifecycleBatch {
            sequence: 2,
            team_id: 1,
            authoritative_tick: 3,
            replica_tick: 3,
            view_epoch: 7,
            events: vec![RenderLifecycleAction::Forget {
                replica_id: 105,
                disclosure_epoch: 1,
            }],
        });

        let stale_actions = bridge.apply(&snapshot(2, vec![entity(105, 1)]));
        assert!(stale_actions.iter().all(|action| !matches!(
            action,
            FilteredSceneAction::UpsertDeterministic(entity) if entity.replica_id == 105
        )));
        assert_eq!(bridge.deterministic_count(), 0);

        bridge.apply(&snapshot(4, vec![entity(131, 1)]));
        assert!(bridge.deterministic_entity(105).is_none());
        assert!(bridge.deterministic_entity(131).is_some());
        assert_eq!(bridge.deterministic_count(), 1);
    }

    #[test]
    fn reset_clears_view_and_old_epoch_events_are_ignored() {
        let mut bridge = FilteredRenderBridge::default();
        bridge.apply_lifecycle(&reset(1, 7));
        bridge.apply(&snapshot(2, vec![entity(42, 3)]));
        bridge.apply_lifecycle(&reset(2, 8));
        assert_eq!(bridge.deterministic_count(), 0);
        bridge.apply(&snapshot(3, vec![entity(43, 4)]));
        bridge.apply_lifecycle(&RenderLifecycleBatch {
            sequence: 10,
            team_id: 1,
            authoritative_tick: 4,
            replica_tick: 4,
            view_epoch: 7,
            events: vec![RenderLifecycleAction::Forget {
                replica_id: 43,
                disclosure_epoch: 4,
            }],
        });
        assert!(bridge.deterministic_entity(43).is_some());
    }

    #[test]
    fn repeated_boundary_crossings_do_not_accumulate_identities() {
        let mut bridge = FilteredRenderBridge::default();
        bridge.apply_lifecycle(&reset(1, 7));
        for replica_id in 100..120 {
            bridge.apply(&snapshot(replica_id, vec![entity(replica_id, 1)]));
            bridge.apply_lifecycle(&RenderLifecycleBatch {
                sequence: replica_id,
                team_id: 1,
                authoritative_tick: replica_id,
                replica_tick: replica_id,
                view_epoch: 7,
                events: vec![RenderLifecycleAction::Forget {
                    replica_id,
                    disclosure_epoch: 1,
                }],
            });
            bridge.apply(&snapshot(
                replica_id.saturating_sub(1),
                vec![entity(replica_id, 1)],
            ));
            assert_eq!(bridge.deterministic_count(), 0);
        }
    }

    #[test]
    fn owned_hero_hud_comes_from_disclosed_components() {
        let hero = Hero::new("hero_test".into(), "測試英雄".into(), "守霧者".into());
        let property = CProperty {
            hp: Fixed64::from_i32(80),
            mhp: Fixed64::from_i32(100),
            msd: Fixed64::from_i32(325),
            def_physic: Fixed64::from_i32(12),
            def_magic: Fixed64::from_i32(9),
        };
        let components = BTreeMap::from([
            (
                DEMO_RENDER_COMPONENT_SCHEMA_ID,
                omoba_core::runtime::encode_demo_render_state(DemoRenderState {
                    x_raw: 0,
                    y_raw: 0,
                    team_id: 1,
                    kind: 1,
                    owner_player_id: 7,
                }),
            ),
            (
                DISCLOSED_HERO_COMPONENT_SCHEMA_ID,
                serde_json::to_vec(&hero).unwrap(),
            ),
            (
                DISCLOSED_PROPERTY_COMPONENT_SCHEMA_ID,
                encode_disclosed_property(&property),
            ),
        ]);
        let mut bridge = FilteredRenderBridge::default();
        bridge.apply(&FilteredRenderSnapshot {
            team_id: 1,
            replica_tick: 10,
            entities: vec![FilteredRenderEntity {
                replica_id: 42,
                disclosure_epoch: 1,
                entity_kind: 1,
                components,
            }],
            public_events: Vec::new(),
            external_effects: Vec::new(),
            memory_directives: Vec::new(),
        });

        let hud = bridge.owned_hero_presentation(7).unwrap();
        assert_eq!(hud.render_id, 42);
        assert_eq!(hud.name, "測試英雄");
        assert_eq!(hud.hp, 80.0);
        assert_eq!(hud.max_hp, 100.0);
        assert_eq!(hud.move_speed, 325.0);
        assert!(bridge.owned_hero_presentation(8).is_none());
    }
}
