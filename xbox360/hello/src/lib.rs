//! A `no_std` Rust test title for the Xbox 360 on xsdk.
//!
//! It proves the target and the link before any of the engine: the screen
//! clears to a colour the controller sets (left stick x = red, right stick
//! y = green, right trigger = blue, left trigger dims), A rumbles, and Back
//! ends it, as xsdk's own `hello` sample does. It reports `PASS:`/`FAIL:`
//! lines in `rust_hello.log` beside the `.xex`.
//!
//! Rust's `main` is the title's: xsdk's start-up calls it directly, and it
//! calls xsdk directly, as the conventions match (xbox360/README.md).

#![no_std]

use xsdk_rt::{Checks, Gamepad, NtStatus, log};

unsafe extern "C" {
    fn xsdk_gpu_open() -> NtStatus;
    fn xsdk_gpu_close() -> NtStatus;
    fn xsdk_present_open(
        mode: u32,
        buffer_count: u32,
        swap_interval: u32,
        width: *mut u32,
        height: *mut u32,
    ) -> NtStatus;
    fn xsdk_present_begin(address: *mut u32, timeout_ms: u32) -> NtStatus;
    fn xsdk_gpu_resolve(address: u32, width: u32, height: u32, clear_argb: u32);
    fn xsdk_present_pixels(address: u32) -> *mut u32;
    fn xsdk_present_end();
    fn xsdk_present_close();
    fn xsdk_gpu_fence() -> u32;
    fn xsdk_gpu_wait(fence: u32, timeout_ms: u32) -> NtStatus;
}

/// `XsdkDisplayMode`'s `XSDK_DISPLAY_720P` (xsdk `include/xsdk/display.h`):
/// the first enumerator.
const DISPLAY_720P: u32 = 0;
const SPLASH_MS: u32 = 2000;
/// The Guide's first opening held a frame up 2.7 s (xsdk docs/HARDWARE.md).
const BEGIN_TIMEOUT_MS: u32 = 5000;
const RUMBLE_SPEED: u16 = 0x8000;
const FENCE_TIMEOUT_MS: u32 = 1000;

/// The clear colour for a controller state, as xsdk's `hello` computes it.
fn colour_of(pad: &Gamepad) -> u32 {
    let dim = 255 - u32::from(pad.left_trigger);
    let red = ((i32::from(pad.thumb_lx) + 32768) as u32) >> 8;
    let green = ((i32::from(pad.thumb_ry) + 32768) as u32) >> 8;
    let blue = u32::from(pad.right_trigger);
    0xFF00_0000 | (red * dim / 255) << 16 | (green * dim / 255) << 8 | (blue * dim / 255)
}

/// Pure-Rust work with a known answer, so the log shows LLVM's code ran
/// right on the console's CPU: 64-bit integers (one register each on this
/// target), floats, and the byte order.
fn self_test(checks: &mut Checks) {
    let mut acc: u64 = 0x0123_4567_89AB_CDEF;
    for i in 0..1000u64 {
        acc = acc.wrapping_mul(6364136223846793005).wrapping_add(i);
    }
    checks.check("rust_hello: 64-bit arithmetic", acc == SELF_TEST_U64);
    let mut sum = 0.0f32;
    for i in 1..=100 {
        sum += 1.0 / i as f32;
    }
    checks.check("rust_hello: float arithmetic", (sum - 5.187_378).abs() < 1e-4);
    checks.check(
        "rust_hello: the CPU is big-endian",
        0x1122_3344u32.to_ne_bytes() == [0x11, 0x22, 0x33, 0x44],
    );
}

include!(concat!(env!("OUT_DIR"), "/self_test.rs"));

fn run(width: u32, height: u32, checks: &mut Checks) {
    let mut last_result = xsdk_rt::ERROR_SUCCESS;
    let mut rumbling = false;
    let mut cleared = 0u32;
    let mut frame: u32 = 0;
    loop {
        let (result, pad) = xsdk_rt::pad(0);
        // Logged when it changes: a missing controller fails every frame.
        if result != last_result {
            log!("rust_hello: XamInputGetState returned 0x{:08X}", result);
        }
        last_result = result;
        if pad.buttons & xsdk_rt::BUTTON_BACK != 0 {
            log!("rust_hello: Back pressed after {} frames", frame);
            break;
        }
        let a = pad.buttons & xsdk_rt::BUTTON_A != 0;
        if a != rumbling {
            let speed = if a { RUMBLE_SPEED } else { 0 };
            let r = xsdk_rt::rumble(0, speed, speed);
            log!("rust_hello: A {}: XamInputSetState({:#06X}) returned 0x{:08X}", if a { "down" } else { "up" }, speed, r);
        }
        rumbling = a;

        let mut address = 0u32;
        let status = unsafe { xsdk_present_begin(&mut address, BEGIN_TIMEOUT_MS) };
        if status < 0 {
            log!("rust_hello: xsdk_present_begin failed: 0x{:08X}", status as u32);
            break;
        }
        let argb = colour_of(&pad);
        unsafe { xsdk_gpu_resolve(address, width, height, argb) };
        // A resolve copies eDRAM, then clears it: frame 1 shows frame 0's clear.
        if frame == 1 {
            // The resolve is queued GPU work: wait for it before reading what it wrote, as xsdk's hello does.
            let status = unsafe { xsdk_gpu_wait(xsdk_gpu_fence(), FENCE_TIMEOUT_MS) };
            if status < 0 {
                log!("rust_hello: xsdk_gpu_wait failed: 0x{:08X}", status as u32);
            }
            // The first pixel of the first tile, where untiling can't move it.
            // Each pixel's bytes are B, G, R, A (xsdk docs/HARDWARE.md).
            let got = unsafe { xsdk_present_pixels(address).read_volatile() }.swap_bytes();
            log!("rust_hello: the first frame cleared to {:08X}, read {:08X}", cleared, got);
            checks.check(
                "rust_hello: the first resolved frame holds the colour it was cleared to",
                got == cleared,
            );
        }
        unsafe { xsdk_present_end() };
        cleared = argb;
        frame = frame.wrapping_add(1);
    }
    if rumbling {
        xsdk_rt::rumble(0, 0, 0);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn main(_argc: i32, _argv: *const *const u8) -> i32 {
    let mut checks = Checks::default();
    log!("rust_hello: a Rust title; move the sticks and triggers, hold A to rumble, Back ends");
    self_test(&mut checks);
    let splash = unsafe { xsdk_rt::xsdk_splash_show(SPLASH_MS) };
    if splash < 0 {
        log!("rust_hello: xsdk_splash_show failed: 0x{:08X}", splash as u32);
    }
    let status = unsafe { xsdk_gpu_open() };
    if status < 0 {
        log!("rust_hello: xsdk_gpu_open failed: 0x{:08X}", status as u32);
        return 1;
    }
    let (mut width, mut height) = (0u32, 0u32);
    let status = unsafe { xsdk_present_open(DISPLAY_720P, 2, 1, &mut width, &mut height) };
    if status < 0 {
        log!("rust_hello: xsdk_present_open failed: 0x{:08X}", status as u32);
        return 1;
    }
    log!("rust_hello: the display is {}x{}", width, height);
    run(width, height, &mut checks);
    unsafe { xsdk_present_close() };
    log!("rust_hello: display closed");
    let status = unsafe { xsdk_gpu_close() };
    if status < 0 {
        log!("rust_hello: xsdk_gpu_close failed: 0x{:08X}", status as u32);
    }
    log!("rust_hello: done");
    checks.finish()
}
