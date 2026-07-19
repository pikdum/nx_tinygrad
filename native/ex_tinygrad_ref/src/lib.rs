//! Tensor-reference resource for ex_tinygrad.
//!
//! This NIF owns only reference *metadata* — a worker id, a generation, and a
//! buffer handle. It never touches tinygrad, Python, KFD, or any GPU API.
//!
//! When a `TensorRef` resource is garbage-collected, its `Drop` impl atomically
//! marks it released and pushes `{worker_id, generation, handle}` onto a global
//! queue. It never blocks on I/O and never talks to the Port.
//! `ExTinygrad.ReleaseReaper` drains the queue and sends batched release requests.

use rustler::{Resource, ResourceArc};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, Mutex};

/// Global queue of releases produced by dropped resources.
static RELEASE_QUEUE: LazyLock<Mutex<Vec<(u64, u64, u64)>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));

pub struct TensorRef {
    worker_id: u64,
    generation: u64,
    handle: u64,
    released: AtomicBool,
}

#[rustler::resource_impl]
impl Resource for TensorRef {}

impl Drop for TensorRef {
    fn drop(&mut self) {
        // If it was already explicitly taken/released, don't queue again.
        if !self.released.swap(true, Ordering::AcqRel) {
            if let Ok(mut queue) = RELEASE_QUEUE.lock() {
                queue.push((self.worker_id, self.generation, self.handle));
            }
        }
    }
}

#[rustler::nif]
fn new(worker_id: u64, generation: u64, handle: u64) -> ResourceArc<TensorRef> {
    ResourceArc::new(TensorRef {
        worker_id,
        generation,
        handle,
        released: AtomicBool::new(false),
    })
}

/// Atomically claim the reference for explicit release. Returns the release
/// tuple the first time, `nil` afterwards, so a later GC cannot double-release.
#[rustler::nif]
fn take(resource: ResourceArc<TensorRef>) -> Option<(u64, u64, u64)> {
    if !resource.released.swap(true, Ordering::AcqRel) {
        Some((resource.worker_id, resource.generation, resource.handle))
    } else {
        None
    }
}

#[rustler::nif]
fn handle(resource: ResourceArc<TensorRef>) -> u64 {
    resource.handle
}

#[rustler::nif]
fn generation(resource: ResourceArc<TensorRef>) -> u64 {
    resource.generation
}

/// Drain and return all queued releases produced by dropped resources.
#[rustler::nif]
fn drain_releases() -> Vec<(u64, u64, u64)> {
    match RELEASE_QUEUE.lock() {
        Ok(mut queue) => std::mem::take(&mut *queue),
        Err(_) => Vec::new(),
    }
}

rustler::init!("Elixir.ExTinygrad.TensorRef");
