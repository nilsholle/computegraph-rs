use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::traits::GraphOperation;

/// Graph-local value identifier.
pub type LocalValueId = usize;

/// Graph-local operation identifier.
pub type LocalOperationId = usize;

/// Describes the role an operation plays in a graph.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum OperationRole {
    Primary,
    Linearized { active_mask: Vec<bool> },
}

/// Reference to a value: either local to the current graph or external.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ValueRef<Op: GraphOperation> {
    Local(LocalValueId),
    External(ValueKey<Op>),
}

/// Cross-graph structural identity for a value.
#[derive(Clone, Debug)]
pub enum ValueKey<Op: GraphOperation> {
    Input(Op::InputKey),
    Derived {
        /// Shared structural identity of the operation that produced this value.
        operation: Arc<OperationKey<Op>>,
        output_slot: u8,
    },
}

/// Cross-graph structural identity for an operation.
///
/// `OperationKey` caches a structural fingerprint so maps keyed by recursively
/// derived values can avoid repeatedly re-hashing the whole input tree. Equality
/// still checks the full structure after the fingerprint prefilter.
#[derive(Debug)]
pub struct OperationKey<Op: GraphOperation> {
    operation: Op,
    inputs: Vec<ValueKey<Op>>,
    role: OperationRole,
    /// Equivalence class of this key: two keys carrying the same class id are
    /// structurally equal.
    ///
    /// A fresh key gets a unique id; a clone inherits it; a structural
    /// comparison that succeeds gives both sides the smaller of the two ids.
    /// Without it, comparing two separately built keys of a shared expression
    /// walks the DAG as a tree (every shared subexpression once per use),
    /// which is exponential in the depth of the sharing.
    class: AtomicU64,
    /// Cached hash prefilter for recursively structural keys.
    ///
    /// This is not an identity proof: equality still compares the full
    /// structure after the fingerprint matches, so hash collisions remain
    /// correct.
    fingerprint: u64,
}

impl<Op: GraphOperation> OperationKey<Op> {
    /// Builds an operation key and precomputes its structural fingerprint.
    ///
    /// # Examples
    ///
    /// ```
    /// use computegraph::{GraphOperation, OperationKey, OperationRole, ValueKey};
    ///
    /// #[derive(Clone, Debug, Hash, PartialEq, Eq)]
    /// enum Op {
    ///     Add,
    /// }
    ///
    /// impl GraphOperation for Op {
    ///     type Operand = f64;
    ///     type Context = ();
    ///     type InputKey = &'static str;
    ///
    ///     fn input_count(&self) -> usize { 2 }
    ///     fn output_count(&self) -> usize { 1 }
    /// }
    ///
    /// let key = OperationKey::new(
    ///     Op::Add,
    ///     vec![ValueKey::Input("x"), ValueKey::Input("y")],
    ///     OperationRole::Primary,
    /// );
    /// assert_eq!(key.inputs().len(), 2);
    /// assert_eq!(key.role(), &OperationRole::Primary);
    /// ```
    pub fn new(operation: Op, inputs: Vec<ValueKey<Op>>, role: OperationRole) -> Self {
        let fingerprint = fingerprint_operation(&operation, &inputs, &role);
        Self {
            operation,
            inputs,
            role,
            class: AtomicU64::new(NEXT_CLASS.fetch_add(1, Ordering::Relaxed)),
            fingerprint,
        }
    }

    /// Returns the cached structural fingerprint.
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    /// Returns the operation.
    pub fn operation(&self) -> &Op {
        &self.operation
    }

    /// Returns the structural input keys.
    pub fn inputs(&self) -> &[ValueKey<Op>] {
        &self.inputs
    }

    /// Returns the role of this operation in the graph.
    pub fn role(&self) -> &OperationRole {
        &self.role
    }
}

impl<Op: GraphOperation> PartialEq for ValueKey<Op> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Input(lhs), Self::Input(rhs)) => lhs == rhs,
            (
                Self::Derived {
                    operation: lhs_op,
                    output_slot: lhs_slot,
                },
                Self::Derived {
                    operation: rhs_op,
                    output_slot: rhs_slot,
                },
            ) => {
                lhs_slot == rhs_slot
                    && (Arc::ptr_eq(lhs_op, rhs_op) || lhs_op.as_ref() == rhs_op.as_ref())
            }
            _ => false,
        }
    }
}

impl<Op: GraphOperation> Eq for ValueKey<Op> {}

impl<Op: GraphOperation> Hash for ValueKey<Op> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Self::Input(key) => {
                0u8.hash(state);
                key.hash(state);
            }
            Self::Derived {
                operation,
                output_slot,
            } => {
                1u8.hash(state);
                operation.fingerprint.hash(state);
                output_slot.hash(state);
            }
        }
    }
}

/// Source of fresh equivalence-class ids (never reused).
static NEXT_CLASS: AtomicU64 = AtomicU64::new(0);

impl<Op: GraphOperation> Clone for OperationKey<Op> {
    fn clone(&self) -> Self {
        Self {
            operation: self.operation.clone(),
            inputs: self.inputs.clone(),
            role: self.role.clone(),
            class: AtomicU64::new(self.class.load(Ordering::Relaxed)),
            fingerprint: self.fingerprint,
        }
    }
}

impl<Op: GraphOperation> PartialEq for OperationKey<Op> {
    fn eq(&self, other: &Self) -> bool {
        if self.fingerprint != other.fingerprint {
            return false;
        }
        let lhs_class = self.class.load(Ordering::Relaxed);
        let rhs_class = other.class.load(Ordering::Relaxed);
        if lhs_class == rhs_class {
            return true;
        }
        let equal = self.operation == other.operation
            && self.role == other.role
            && self.inputs == other.inputs;
        if equal {
            // Every id a key ever carries belongs to a key proven equal to it,
            // so a racing store can only replace one valid id by another.
            let class = lhs_class.min(rhs_class);
            self.class.store(class, Ordering::Relaxed);
            other.class.store(class, Ordering::Relaxed);
        }
        equal
    }
}

impl<Op: GraphOperation> Eq for OperationKey<Op> {}

impl<Op: GraphOperation> Hash for OperationKey<Op> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.fingerprint.hash(state);
    }
}

fn fingerprint_operation<Op: GraphOperation>(
    operation: &Op,
    inputs: &[ValueKey<Op>],
    role: &OperationRole,
) -> u64 {
    let mut hasher = DefaultHasher::new();
    operation.hash(&mut hasher);
    role.hash(&mut hasher);
    inputs.len().hash(&mut hasher);
    for input in inputs {
        fingerprint_val(input).hash(&mut hasher);
    }
    hasher.finish()
}

fn fingerprint_val<Op: GraphOperation>(key: &ValueKey<Op>) -> u64 {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}
