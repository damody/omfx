//! Presentation-only bridge for the disclosed replica world. Remembered
//! ghosts are deliberately stored outside target, collision, simulation and
//! hash state.

use std::collections::{BTreeMap, BTreeSet};

use omoba_core::runtime::{FilteredRenderEntity, FilteredRenderSnapshot, RenderMemoryDirective};

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
pub struct RememberPresentationRule { pub ttl_ticks: Option<u64> }

#[derive(Clone, Debug, Default)]
pub struct RememberPresentationRegistry { rules: BTreeMap<u32, RememberPresentationRule> }

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
        registry.register(1, RememberPresentationRule { ttl_ticks: Some(120 * 10) });
        Self { deterministic: BTreeMap::new(), remembered: BTreeMap::new(), retired: BTreeSet::new(), registry }
    }
}

impl FilteredRenderBridge {
    pub fn deterministic_count(&self) -> usize { self.deterministic.len() }

    pub fn remembered_count(&self) -> usize { self.remembered.len() }

    pub fn deterministic_demo_states(&self) -> impl Iterator<Item = (u64, omoba_core::runtime::DemoRenderState)> + '_ {
        self.deterministic.iter().filter_map(|(id, entity)| {
            entity.components.get(&omoba_core::runtime::DEMO_RENDER_COMPONENT_SCHEMA_ID)
                .and_then(|bytes| omoba_core::runtime::decode_demo_render_state(bytes))
                .map(|state| (*id, state))
        })
    }

    pub fn remembered_demo_states(&self) -> impl Iterator<Item = (RememberedAssociationKey, omoba_core::runtime::DemoRenderState)> + '_ {
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
                RenderMemoryDirective::Hide { replica_id, disclosure_epoch, remember_policy, sanitized_presentation } => {
                    self.deterministic.remove(replica_id);
                    actions.push(FilteredSceneAction::RemoveDeterministicNode(*replica_id));
                    if let Some(rule) = self.registry.lookup(*remember_policy) {
                        let presentation = RememberedPresentation {
                            key: RememberedAssociationKey { replica_id: *replica_id, disclosure_epoch: *disclosure_epoch },
                            presentation_kind: *remember_policy,
                            sanitized_payload: sanitized_presentation.clone(),
                            expires_at_tick: rule.ttl_ticks.map(|ttl| snapshot.replica_tick.saturating_add(ttl)),
                        };
                        self.remembered.insert(presentation.key, presentation.clone());
                        actions.push(FilteredSceneAction::InsertRemembered(presentation));
                    }
                }
                RenderMemoryDirective::Forget { replica_id, disclosure_epoch } => {
                    self.deterministic.remove(replica_id);
                    self.retired.insert(*replica_id);
                    let key = RememberedAssociationKey { replica_id: *replica_id, disclosure_epoch: *disclosure_epoch };
                    self.remembered.remove(&key);
                    actions.push(FilteredSceneAction::RemoveDeterministicNode(*replica_id));
                    actions.push(FilteredSceneAction::RemoveRemembered(key));
                    actions.push(FilteredSceneAction::RetireDeterministicIdentity(*replica_id));
                }
            }
        }
        for entity in &snapshot.entities {
            let key = RememberedAssociationKey { replica_id: entity.replica_id, disclosure_epoch: entity.disclosure_epoch };
            if self.remembered.remove(&key).is_some() {
                actions.push(FilteredSceneAction::RemoveRemembered(key));
            }
            self.deterministic.insert(entity.replica_id, entity.clone());
            actions.push(FilteredSceneAction::UpsertDeterministic(entity.clone()));
        }
        actions
    }

    fn expire(&mut self, tick: u64, actions: &mut Vec<FilteredSceneAction>) {
        let expired: Vec<_> = self.remembered.iter()
            .filter(|(_, value)| value.expires_at_tick.is_some_and(|expiry| expiry <= tick))
            .map(|(key, _)| *key).collect();
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
    pub fn remembered_presentation(&self, key: RememberedAssociationKey) -> Option<&RememberedPresentation> {
        self.remembered.get(&key)
    }
}
