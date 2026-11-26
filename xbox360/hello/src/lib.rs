//! A `no_std` Rust test title for the Xbox 360 on xsdk.
//!
//! It proves the custom target and the link before any of the engine: the
//! screen clears to a colour the controller sets (left stick x = red, right
//! stick y = green, right trigger = blue, left trigger dims), A rumbles, and
//! Back ends it, as xsdk's own `hello` sample does. It checks itself the way
//! xsdk's samples do - `PASS:`/`FAIL:` lines in the log beside the `.xex` -
//! since nothing here can see the screen.
//!
//! Everything this crate calls is reached through `bridge.c`'s thunks, and
//! only `rust_title_main` is called from C (xbox360/README.md, "The calling
//! convention boundary"). Keep every argument and return value across it to
//! `i32`/`u32`/pointers: no `u8`/`u16`/`bool` returns and no 64-bit integers.

#![no_std]

use core::panic::PanicInfo;

mod xsdk {
    /// `INPUT_GAMEPAD` (xsdk `include/xsdk/kernel.h`). The console is
    /// big-endian like this code, so the layout matches with no swapping.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct Gamepad {
        pub buttons: u16,
        pub left_trigger: u8,
        pub right_trigger: u8,
        pub thumb_lx: i16,
        pub thumb_ly: i16,
        pub thumb_rx: i16,
        pub thumb_ry: i16,
    }

    /// `INPUT_STATE`.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    pub struct InputState {
        pub packet_number: u32,
        pub gamepad: Gamepad,
    }

    pub const BUTTON_BACK: u16 = 0x0020;
    pub const BUTTON_A: u16 = 0x1000;
    pub const ERROR_SUCCESS: u32 = 0;

    unsafe extern "C" {
        // bridge.c's own helpers (xsdk_log_write is varargs, the display
        // mode an enum: neither crosses the boundary directly).
        pub fn bridge_log(text: *const u8, length: u32);
        pub fn bridge_present_open(width: *mut u32, height: *mut u32) -> i32;
        pub fn bridge_set_rumble(user: u32, low: u32, high: u32) -> u32;
        // xsdk itself, unchanged.
        pub fn xsdk_splash_show(duration_ms: u32) -> i32;
        pub fn xsdk_gpu_open() -> i32;
        pub fn xsdk_gpu_close() -> i32;
        pub fn xsdk_present_begin(address: *mut u32, timeout_ms: u32) -> i32;
        pub fn xsdk_gpu_resolve(address: u32, width: u32, height: u32, clear_argb: u32);
        pub fn xsdk_present_pixels(address: u32) -> *mut u32;
        pub fn xsdk_present_end();
        pub fn xsdk_present_close();
        pub fn XamInputGetState(user: u32, flags: u32, state: *mut InputState) -> u32;
    }
}

const SPLASH_MS: u32 = 2000;
/// The Guide's first opening held a frame up 2.7 s (xsdk docs/HARDWARE.md).
const BEGIN_TIMEOUT_MS: u32 = 5000;
const RUMBLE_SPEED: u32 = 0x8000;

/// A fixed-size line for the log: `core::fmt` without an allocator.
struct Line {
    bytes: [u8; 160],
    len: usize,
}

impl Line {
    fn new() -> Line {
        Line {
            bytes: [0; 160],
            len: 0,
        }
    }
}

impl core::fmt::Write for Line {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let room = self.bytes.len() - self.len;
        let take = s.len().min(room);
        self.bytes[self.len..self.len + take].copy_from_slice(&s.as_bytes()[..take]);
        self.len += take;
        Ok(())
    }
}

fn log_line(text: &str) {
    unsafe { xsdk::bridge_log(text.as_ptr(), text.len() as u32) };
}

macro_rules! log {
    ($($arg:tt)*) => {{
        let mut line = Line::new();
        let _ = core::fmt::Write::write_fmt(&mut line, format_args!($($arg)*));
        unsafe { xsdk::bridge_log(line.bytes.as_ptr(), line.len as u32) };
    }};
}

#[derive(Default)]
struct Checks {
    passed: u32,
    failed: u32,
}

impl Checks {
    fn check(&mut self, name: &str, passed: bool) {
        log!("{}: {}", if passed { "PASS" } else { "FAIL" }, name);
        if passed {
            self.passed += 1;
        } else {
            self.failed += 1;
        }
    }
}

/// The clear colour for a controller state, as xsdk's `hello` computes it.
fn colour_of(pad: &xsdk::Gamepad) -> u32 {
    let dim = 255 - u32::from(pad.left_trigger);
    let red = ((i32::from(pad.thumb_lx) + 32768) as u32) >> 8;
    let green = ((i32::from(pad.thumb_ry) + 32768) as u32) >> 8;
    let blue = u32::from(pad.right_trigger);
    0xFF00_0000 | (red * dim / 255) << 16 | (green * dim / 255) << 8 | (blue * dim / 255)
}

fn set_rumble(on: bool) {
    let speed = if on { RUMBLE_SPEED } else { 0 };
    let result = unsafe { xsdk::bridge_set_rumble(0, speed, speed) };
    if result != xsdk::ERROR_SUCCESS {
        log!("rust_hello: XamInputSetState failed: 0x{:08X}", result);
    }
}

/// Some pure-Rust work whose answer is known, so the log shows LLVM's code
/// (64-bit arithmetic through compiler-builtins, floats, slices) ran right
/// on the console's CPU, not just that the calls went through.
fn self_test(checks: &mut Checks) {
    let mut acc: u64 = 0x0123_4567_89AB_CDEF;
    for i in 0..1000u64 {
        acc = acc.wrapping_mul(6364136223846793005).wrapping_add(i);
    }
    // Reference value from the same loop on the PC.
    checks.check("rust_hello: 64-bit arithmetic", acc == SELF_TEST_U64);
    let mut sum = 0.0f32;
    for i in 1..=100 {
        sum += 1.0 / i as f32;
    }
    checks.check(
        "rust_hello: float arithmetic",
        (sum - 5.187_378).abs() < 1e-4,
    );
    let word = 0x1122_3344u32;
    checks.check(
        "rust_hello: the CPU is big-endian",
        word.to_ne_bytes() == [0x11, 0x22, 0x33, 0x44],
    );
}

include!(concat!(env!("OUT_DIR"), "/self_test.rs"));

fn run(width: u32, height: u32, checks: &mut Checks) {
    let mut last_result = xsdk::ERROR_SUCCESS;
    let mut rumbling = false;
    let mut cleared = 0u32;
    let mut frame: u32 = 0;
    loop {
        let mut input = xsdk::InputState::default();
        let result = unsafe { xsdk::XamInputGetState(0, 0, &mut input) };
        // Logged when it changes: a missing controller fails every frame.
        if result != last_result {
            log!("rust_hello: XamInputGetState returned 0x{:08X}", result);
        }
        last_result = result;
        if result != xsdk::ERROR_SUCCESS {
            input = xsdk::InputState::default();
        }
        if input.gamepad.buttons & xsdk::BUTTON_BACK != 0 {
            break;
        }
        let a = input.gamepad.buttons & xsdk::BUTTON_A != 0;
        if a != rumbling {
            set_rumble(a);
        }
        rumbling = a;

        let mut address = 0u32;
        let status = unsafe { xsdk::xsdk_present_begin(&mut address, BEGIN_TIMEOUT_MS) };
        if status < 0 {
            log!("rust_hello: xsdk_present_begin failed: 0x{:08X}", status as u32);
            break;
        }
        let argb = colour_of(&input.gamepad);
        unsafe { xsdk::xsdk_gpu_resolve(address, width, height, argb) };
        // A resolve copies eDRAM then clears it: frame 1 shows frame 0's clear.
        if frame == 1 {
            check_first_frame(address, cleared, checks);
        }
        unsafe { xsdk::xsdk_present_end() };
        cleared = argb;
        frame = frame.wrapping_add(1);
    }
    if rumbling {
        set_rumble(false);
    }
}

/// The resolved frame is tiled, so read a pixel the untiling cannot move: the
/// first one of the first tile. Each pixel's bytes are B, G, R, A in memory
/// (xsdk docs/HARDWARE.md, "A title's buffer on screen").
fn check_first_frame(address: u32, argb: u32, checks: &mut Checks) {
    let pixels = unsafe { xsdk::xsdk_present_pixels(address) };
    // Volatile: the GPU wrote it, the compiler cannot know.
    let raw = unsafe { pixels.read_volatile() };
    let got = raw.swap_bytes();
    log!("rust_hello: the first frame cleared to {:08X}, read {:08X}", argb, got);
    checks.check(
        "rust_hello: the first resolved frame holds the colour it was cleared to",
        got == argb,
    );
}

/// The title, called from `bridge.c`'s `main` through its entry thunk.
#[unsafe(no_mangle)]
pub extern "C" fn rust_title_main() -> i32 {
    let mut checks = Checks::default();
    log_line("rust_hello: a Rust title; move the sticks and triggers, hold A to rumble, Back ends");
    self_test(&mut checks);
    let splash = unsafe { xsdk::xsdk_splash_show(SPLASH_MS) };
    if splash < 0 {
        log!("rust_hello: xsdk_splash_show failed: 0x{:08X}", splash as u32);
    }
    let status = unsafe { xsdk::xsdk_gpu_open() };
    if status < 0 {
        log!("rust_hello: xsdk_gpu_open failed: 0x{:08X}", status as u32);
        return 1;
    }
    let (mut width, mut height) = (0u32, 0u32);
    let status = unsafe { xsdk::bridge_present_open(&mut width, &mut height) };
    if status < 0 {
        log!("rust_hello: xsdk_present_open failed: 0x{:08X}", status as u32);
        return 1;
    }
    log!("rust_hello: the display is {}x{}", width, height);
    run(width, height, &mut checks);
    unsafe { xsdk::xsdk_present_close() };
    let status = unsafe { xsdk::xsdk_gpu_close() };
    if status < 0 {
        log!("rust_hello: xsdk_gpu_close failed: 0x{:08X}", status as u32);
    }
    log_line("rust_hello: Back pressed, done");
    log!("checks: {} passed, {} failed", checks.passed, checks.failed);
    if checks.failed == 0 { 0 } else { 2 }
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    log!("rust_hello: panic: {}", info.message());
    unsafe extern "C" {
        fn bridge_abort() -> !;
    }
    unsafe { bridge_abort() }
}
