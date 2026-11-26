//! What alpha3's `SDLActivity` calls into `libshock2.so`.
//!
//! * `SDL_main`: SDLActivity's `nativeRunMain` looks it up in the main
//!   library and runs the game on SDL's thread.
//! * `SDLActivity.isMouseShown()` / `getMouseX()` / `getMouseY()`: natives
//!   the launcher polls every frame (EngineActivity's AutoMouseModeComposable)
//!   to show its cursor and switch its touch controls to mouse mode while a
//!   menu, use mode or an MFD panel is up.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::io::{BufRead, BufReader, Write};
use std::os::fd::FromRawFd;

use jni_sys::{JNIEnv, JavaVM, jclass, jint};

#[unsafe(no_mangle)]
pub extern "C" fn SDL_main(argc: c_int, argv: *const *const c_char) -> c_int {
    redirect_stdio_to_logcat();
    init_ndk_context();
    let args: Vec<String> = (0..argc.max(0) as usize)
        .map(|i| {
            unsafe { CStr::from_ptr(*argv.add(i)) }
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    std::panic::catch_unwind(|| crate::run(args)).unwrap_or(1)
}

#[unsafe(no_mangle)]
pub extern "C" fn Java_org_libsdl_app_SDLActivity_isMouseShown(
    _env: *mut JNIEnv,
    _class: jclass,
) -> jint {
    jint::from(crate::pointer_shown())
}

#[unsafe(no_mangle)]
pub extern "C" fn Java_org_libsdl_app_SDLActivity_getMouseX(
    _env: *mut JNIEnv,
    _class: jclass,
) -> jint {
    crate::pointer_position().0
}

#[unsafe(no_mangle)]
pub extern "C" fn Java_org_libsdl_app_SDLActivity_getMouseY(
    _env: *mut JNIEnv,
    _class: jclass,
) -> jint {
    crate::pointer_position().1
}

/// Before the game reads any path: create the save root, and point the
/// engine's own bundled files at `<game folder>/assets` unless the host set
/// `SHOCK2_BUNDLE_ASSETS` (there is no working directory to find `./assets`).
pub fn prepare_paths() {
    let root = shock2vr::paths::data_root();
    println!("shock2: game folder {}", root.display());
    if std::env::var_os("SHOCK2_BUNDLE_ASSETS").is_none() {
        // SAFETY: called on SDL's main thread before the game starts any
        // thread of its own that could read the environment.
        unsafe { std::env::set_var("SHOCK2_BUNDLE_ASSETS", root.join("assets")) };
    }
    shock2vr::paths::init_android_save_root();
}

unsafe extern "C" {
    fn SDL_AndroidGetJNIEnv() -> *mut c_void;
    fn SDL_AndroidGetActivity() -> *mut c_void;
}

/// cpal (audio, through rodio) and anything else built on `ndk-context`
/// expect a JavaVM and an Android Context; under SDLActivity nobody else
/// provides them, so hand over SDL's.
fn init_ndk_context() {
    unsafe {
        let env = SDL_AndroidGetJNIEnv() as *mut JNIEnv;
        if env.is_null() {
            eprintln!("shock2: no JNIEnv from SDL");
            return;
        }
        let mut vm: *mut JavaVM = std::ptr::null_mut();
        let Some(get_java_vm) = (**env).GetJavaVM else {
            return;
        };
        if get_java_vm(env, &mut vm) != 0 || vm.is_null() {
            eprintln!("shock2: GetJavaVM failed");
            return;
        }
        let activity = SDL_AndroidGetActivity() as jni_sys::jobject;
        if activity.is_null() {
            eprintln!("shock2: no activity from SDL");
            return;
        }
        let (Some(new_global), Some(delete_local)) = ((**env).NewGlobalRef, (**env).DeleteLocalRef)
        else {
            return;
        };
        let context = new_global(env, activity);
        delete_local(env, activity);
        ndk_context::initialize_android_context(vm.cast(), context.cast());
    }
}

#[link(name = "log")]
unsafe extern "C" {
    fn __android_log_write(priority: c_int, tag: *const c_char, text: *const c_char) -> c_int;
}

const ANDROID_LOG_INFO: c_int = 4;

/// Send stdout and stderr (println!, the tracing subscriber, panics) to
/// logcat under the tag `shock2`, and to `shock2.log` in the game folder for
/// players without adb.
fn redirect_stdio_to_logcat() {
    let mut fds = [0 as c_int; 2];
    unsafe {
        if libc_pipe(fds.as_mut_ptr()) != 0 {
            return;
        }
        libc_dup2(fds[1], 1);
        libc_dup2(fds[1], 2);
    }
    let reader = unsafe { std::fs::File::from_raw_fd(fds[0]) };
    let _ = std::thread::Builder::new()
        .name("shock2-log".into())
        .spawn(move || {
            let mut file =
                std::fs::File::create(shock2vr::paths::data_root().join("shock2.log")).ok();
            let tag = c"shock2";
            for line in BufReader::new(reader).lines().map_while(Result::ok) {
                if let Some(f) = file.as_mut() {
                    let _ = writeln!(f, "{line}");
                }
                if let Ok(text) = std::ffi::CString::new(line) {
                    unsafe { __android_log_write(ANDROID_LOG_INFO, tag.as_ptr(), text.as_ptr()) };
                }
            }
        });
}

unsafe extern "C" {
    #[link_name = "pipe"]
    fn libc_pipe(fds: *mut c_int) -> c_int;
    #[link_name = "dup2"]
    fn libc_dup2(old: c_int, new: c_int) -> c_int;
}
