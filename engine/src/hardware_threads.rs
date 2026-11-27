//! Which of the Xbox 360's six hardware threads (three cores, two each) a
//! worker runs on. Its kernel otherwise leaves a new thread where it likes,
//! which can be the main thread's own. A no-op everywhere else.

// The main thread runs on hardware thread 0 (core 0), with the kernel's and
// xsdk's own threads around it; the title's workers go on the other cores.

/// The visibility worker (shock2vr's portal_visibility_engine).
pub const VISIBILITY: u32 = 2;
/// The AI pathfinding worker (shock2vr's pathfinding::async_queries).
pub const PATHFINDING: u32 = 3;
/// The audio feeder, which mixes every playing sound (audio::output).
pub const AUDIO: u32 = 4;
/// Level and save parsing behind the loading screen (shock2vr's App).
pub const LOADING: u32 = 5;

#[cfg(target_os = "xbox360")]
unsafe extern "C" {
    fn xsdk_thread_move(hardware_thread: u32) -> i32;
    fn xsdk_thread_hardware_thread() -> u32;
    fn xsdk_thread_current_object() -> u32;
}

/// A value one thread owns, as a `thread_local!` gives each thread its own,
/// without a TLS lookup: the Xbox 360 target has no native thread locals
/// (`has-thread-local: false`), so each `thread_local!` access is a kernel
/// TLS call, which the backend's every uniform write paid three of. The
/// first thread to [`claim`](Self::claim) it owns it; any other sees `None`,
/// as a `thread_local!` that thread never set would.
#[cfg(target_os = "xbox360")]
pub struct OwnedCell<T> {
    owner: std::sync::atomic::AtomicU32,
    value: std::cell::RefCell<T>,
}

// Only the owning thread touches `value` (with and claim check it).
#[cfg(target_os = "xbox360")]
unsafe impl<T> Sync for OwnedCell<T> {}

#[cfg(target_os = "xbox360")]
impl<T> OwnedCell<T> {
    pub const fn new(value: T) -> Self {
        Self {
            owner: std::sync::atomic::AtomicU32::new(0),
            value: std::cell::RefCell::new(value),
        }
    }

    /// Make the calling thread the owner, unless another already is.
    pub fn claim(&self) -> bool {
        let me = unsafe { xsdk_thread_current_object() };
        use std::sync::atomic::Ordering;
        match self
            .owner
            .compare_exchange(0, me, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => true,
            Err(owner) => owner == me,
        }
    }

    /// `f` on the cell itself, from its owner, as `LocalKey::try_with`
    /// (borrowing is `f`'s to try); an error from any other thread.
    pub fn try_with<R>(&self, f: impl FnOnce(&std::cell::RefCell<T>) -> R) -> Result<R, ()> {
        let me = unsafe { xsdk_thread_current_object() };
        if self.owner.load(std::sync::atomic::Ordering::Relaxed) == me {
            Ok(f(&self.value))
        } else {
            Err(())
        }
    }

    /// `f` on the value, from its owner; `None` from any other thread.
    pub fn with<R>(&self, f: impl FnOnce(&mut T) -> R) -> Option<R> {
        let me = unsafe { xsdk_thread_current_object() };
        (self.owner.load(std::sync::atomic::Ordering::Relaxed) == me)
            .then(|| f(&mut self.value.borrow_mut()))
    }
}

/// Move the calling thread to `hardware_thread`.
pub fn place_current(hardware_thread: u32) {
    #[cfg(target_os = "xbox360")]
    {
        let from = unsafe { xsdk_thread_hardware_thread() };
        let status = unsafe { xsdk_thread_move(hardware_thread) };
        tracing::info!(
            "thread {:?}: hardware thread {from} -> {hardware_thread} (status {status:#x})",
            std::thread::current().name()
        );
    }
    #[cfg(not(target_os = "xbox360"))]
    let _ = hardware_thread;
}

/// The calling thread's hardware thread, on the Xbox 360.
pub fn current() -> Option<u32> {
    #[cfg(target_os = "xbox360")]
    return Some(unsafe { xsdk_thread_hardware_thread() });
    #[cfg(not(target_os = "xbox360"))]
    None
}
