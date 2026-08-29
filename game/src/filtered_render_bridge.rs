//! Presentation-only bridge for the disclosed replica world. Remembered
//! ghosts are deliberately stored outside target, collision, simulation and
//! hash state.

use std::collections::{BTreeMap, BTreeSet};

use omoba_core::runtime::{FilteredRenderEntity, FilteredRenderSnapshot, RenderMemoryDirective};

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

#[derive(Debug)]
pub struct FilteredRenderBridge {
    deterministic: BTreeMap<u64, FilteredRenderEntity>,
    remembered: BTreeMap<RememberedAssociationKey, RememberedPresentation>,
    retired: BTreeSet<u64>,
    registry: RememberPresentationRegistry,
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
            retired: BTreeSet::new(),
            registry,
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
        self.expire(snapshot.replica_tick, &mut actions);
        for directive in &snapshot.memory_directives {
            match directive {
                RenderMemoryDirective::Hide {
                    replica_id,
                    disclosure_epoch,
                    remember_policy,
                    sanitized_presentation,
                } => {
                    self.deterministic.remove(replica_id);
                    actions.push(FilteredSceneAction::RemoveDeterministicNode(*replica_id));
                    if let Some(rule) = self.registry.lookup(*remember_policy) {
                        let presentation = RememberedPresentation {
                            key: RememberedAssociationKey {
                                replica_id: *replica_id,
                                disclosure_epoch: *disclosure_epoch,
                            },
                            presentation_kind: *remember_policy,
                            sanitized_payload: sanitized_presentation.clone(),
                            expires_at_tick: rule
                                .ttl_ticks
                                .map(|ttl| snapshot.replica_tick.saturating_add(ttl)),
                        };
                        self.remembered
                            .insert(presentation.key, presentation.clone());
                        actions.push(FilteredSceneAction::InsertRemembered(presentation));
                    }
                }
                RenderMemoryDirective::Forget {
                    replica_id,
                    disclosure_epoch,
                } => {
                    self.deterministic.remove(replica_id);
                    self.retired.insert(*replica_id);
                    let key = RememberedAssociationKey {
                        replica_id: *replica_id,
                        disclosure_epoch: *disclosure_epoch,
                    };
                    self.remembered.remove(&key);
                    actions.push(FilteredSceneAction::RemoveDeterministicNode(*replica_id));
                    actions.push(FilteredSceneAction::RemoveRemembered(key));
                    actions.push(FilteredSceneAction::RetireDeterministicIdentity(
                        *replica_id,
                    ));
                }
            }
        }
        for entity in &snapshot.entities {
            let key = RememberedAssociationKey {
                replica_id: entity.replica_id,
                disclosure_epoch: entity.disclosure_epoch,
            };
            if self.remembered.remove(&key).is_some() {
                actions.push(FilteredSceneAction::RemoveRemembered(key));
            }
            self.deterministic.insert(entity.replica_id, entity.clone());
            actions.push(FilteredSceneAction::UpsertDeterministic(entity.clone()));
        }
        actions
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
