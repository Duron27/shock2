//! What the Rust test titles share on xsdk: the log, self-checks, the
//! controller, a heap over newlib's allocator and the panic handler.
//!
//! The heap and the panic handler are the `runtime` feature, on by default:
//! a title built with std leaves it off and has std's.
//!
//! These are plain `extern "C"` calls into xsdk's C. They are sound because
//! the titles are compiled by xenon-forge's PPC64-ILP32 LLVM, whose calling
//! convention is the console's own, the one xsdk's GCC (`-mcall-xsdk`) uses
//! too (xbox360/README.md, "The calling convention").

#![no_std]

#[cfg(feature = "runtime")]
use core::alloc::{GlobalAlloc, Layout};
use core::ffi::c_char;
#[cfg(feature = "runtime")]
use core::ffi::c_void;
#[cfg(feature = "runtime")]
use core::panic::PanicInfo;

pub type NtStatus = i32;

/// `INPUT_GAMEPAD` (xsdk `include/xsdk/kernel.h`). The console is big-endian
/// like this code, so the layout matches as it is.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct Gamepad {
    pub buttons: u16,
    pub left_trigger: u8,
    pub right_trigger: u8,
    pub thumb_lx: i16,
    pub thumb_ly: i16,
    pub thumb_rx: i16,
    pub thumb_ry: i16,
}

#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct InputState {
    pub packet_number: u32,
    pub gamepad: Gamepad,
}

#[repr(C)]
pub struct InputVibration {
    pub left_motor_speed: u16,
    pub right_motor_speed: u16,
}

pub const BUTTON_BACK: u16 = 0x0020;
pub const BUTTON_A: u16 = 0x1000;
pub const ERROR_SUCCESS: u32 = 0;

unsafe extern "C" {
    pub fn xsdk_log_write(format: *const c_char, ...);
    pub fn xsdk_splash_show(duration_ms: u32) -> NtStatus;
    pub fn XamInputGetState(user: u32, flags: u32, state: *mut InputState) -> u32;
    pub fn XamInputSetState(user: u32, flags: u32, vibration: *const InputVibration) -> u32;
    #[cfg(feature = "runtime")]
    fn memalign(align: usize, size: usize) -> *mut c_void;
    #[cfg(feature = "runtime")]
    fn free(ptr: *mut c_void);
    /// newlib's, which writes xsdk's last log lines.
    pub fn exit(status: i32) -> !;
}

/// A fixed-size log line: `core::fmt` with no allocation, so it works in the
/// panic handler too.
pub struct Line {
    bytes: [u8; 240],
    len: usize,
}

impl Line {
    pub const fn new() -> Line {
        Line {
            bytes: [0; 240],
            len: 0,
        }
    }

    pub fn write(&self) {
        // "%.*s": the line has no terminating zero.
        unsafe {
            xsdk_log_write(
                c"%.*s".as_ptr(),
                self.len as i32,
                self.bytes.as_ptr() as *const c_char,
            )
        };
    }
}

impl Default for Line {
    fn default() -> Self {
        Line::new()
    }
}

impl core::fmt::Write for Line {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let take = s.len().min(self.bytes.len() - self.len);
        self.bytes[self.len..self.len + take].copy_from_slice(&s.as_bytes()[..take]);
        self.len += take;
        Ok(())
    }
}

/// `log!("...", args)`: one line in the title's log beside its `.xex`.
#[macro_export]
macro_rules! log {
    ($($arg:tt)*) => {{
        let mut line = $crate::Line::new();
        let _ = core::fmt::Write::write_fmt(&mut line, format_args!($($arg)*));
        line.write();
    }};
}

/// `PASS:`/`FAIL:` lines and their count, as xsdk's samples report, so a
/// title says from its log what worked without anyone seeing the screen.
#[derive(Default)]
pub struct Checks {
    pub passed: u32,
    pub failed: u32,
}

impl Checks {
    pub fn check(&mut self, name: &str, passed: bool) -> bool {
        log!("{}: {}", if passed { "PASS" } else { "FAIL" }, name);
        if passed {
            self.passed += 1;
        } else {
            self.failed += 1;
        }
        passed
    }

    pub fn finish(&self) -> i32 {
        log!("checks: {} passed, {} failed", self.passed, self.failed);
        if self.failed == 0 { 0 } else { 2 }
    }
}

/// Controller 1 this frame, or all zero when it isn't there (logged on
/// change by the caller if it wants).
pub fn pad(user: u32) -> (u32, Gamepad) {
    let mut state = InputState::default();
    let result = unsafe { XamInputGetState(user, 0, &mut state) };
    if result == ERROR_SUCCESS {
        (result, state.gamepad)
    } else {
        (result, Gamepad::default())
    }
}

pub fn rumble(user: u32, low: u16, high: u16) -> u32 {
    let vibration = InputVibration {
        left_motor_speed: low,
        right_motor_speed: high,
    };
    unsafe { XamInputSetState(user, 0, &vibration) }
}

#[cfg(feature = "runtime")]
/// The heap: newlib's, which xsdk makes thread-safe.
pub struct Heap;

#[cfg(feature = "runtime")]
unsafe impl GlobalAlloc for Heap {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe { memalign(layout.align().max(8), layout.size()) as *mut u8 }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        unsafe { free(ptr as *mut c_void) }
    }
}

#[cfg(feature = "runtime")]
#[global_allocator]
static HEAP: Heap = Heap;

#[cfg(feature = "runtime")]
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    if let Some(location) = info.location() {
        log!(
            "panic at {}:{}: {}",
            location.file(),
            location.line(),
            info.message()
        );
    } else {
        log!("panic: {}", info.message());
    }
    // exit writes xsdk's last log lines.
    unsafe { exit(3) }
}
