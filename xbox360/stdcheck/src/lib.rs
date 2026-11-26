//! Rust's standard library on the Xbox 360: xenon-forge's std backend
//! (`toolchain/rust-std-xbox360`) inside an xsdk title.
//!
//! It checks what the engine leans on: threads, locks, channels,
//! `thread_local!`, the clocks, the heap, `HashMap`'s random keys and files,
//! and a std thread calling xsdk's C. It shows nothing on screen and ends by
//! itself; `rust_stdcheck.log` beside the `.xex` has a `PASS:`/`FAIL:` line
//! for each.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::time::{Duration, Instant, SystemTime};

use xsdk_rt::{Checks, log};

const DIR: &str = r"game:\rust_stdcheck";
const THREADS: u32 = 4;
const PER_THREAD: u32 = 100_000;

thread_local! {
    static COUNT: Cell<u32> = const { Cell::new(0) };
}

fn threads(checks: &mut Checks) {
    let start = Instant::now();
    let total = Arc::new(Mutex::new(0u64));
    let racy = Arc::new(AtomicU32::new(0));
    let handles: Vec<_> = (0..THREADS)
        .map(|n| {
            let total = Arc::clone(&total);
            let racy = Arc::clone(&racy);
            std::thread::Builder::new()
                .name(format!("worker{n}"))
                .spawn(move || {
                    for i in 0..PER_THREAD {
                        *total.lock().unwrap() += u64::from(i);
                        racy.fetch_add(1, Ordering::Relaxed);
                        COUNT.with(|c| c.set(c.get() + 1));
                    }
                    // Each thread's own thread_local, and xsdk's C called from a std thread.
                    let mine = COUNT.with(Cell::get);
                    log!("rust_stdcheck: {} counted {} on its own", std::thread::current().name().unwrap_or("?"), mine);
                    mine
                })
                .expect("spawn")
        })
        .collect();
    let counts: Vec<u32> = handles.into_iter().map(|h| h.join().unwrap_or(0)).collect();
    let expected = u64::from(THREADS) * (u64::from(PER_THREAD) * u64::from(PER_THREAD - 1) / 2);
    let got = *total.lock().unwrap();
    log!("rust_stdcheck: {} threads in {:?}: total {}, atomic {}", THREADS, start.elapsed(), got, racy.load(Ordering::Relaxed));
    checks.check("rust_stdcheck: std::thread spawns and joins, and Mutex keeps the sum", got == expected);
    checks.check("rust_stdcheck: atomics lose no increment", racy.load(Ordering::Relaxed) == THREADS * PER_THREAD);
    checks.check(
        "rust_stdcheck: thread_local! is per thread",
        counts.iter().all(|&c| c == PER_THREAD) && COUNT.with(Cell::get) == 0,
    );
    checks.check(
        "rust_stdcheck: available_parallelism",
        std::thread::available_parallelism().map(|n| {
            log!("rust_stdcheck: available_parallelism {}", n);
            n.get() >= 1
        }).unwrap_or(false),
    );
}

fn sync(checks: &mut Checks) {
    // Condvar: a ping-pong of 1000 round trips.
    let pair = Arc::new((Mutex::new(0u32), Condvar::new()));
    let other = Arc::clone(&pair);
    let start = Instant::now();
    let pong = std::thread::spawn(move || {
        let (lock, cvar) = &*other;
        let mut n = lock.lock().unwrap();
        while *n < 2000 {
            if *n % 2 == 1 {
                *n += 1;
                cvar.notify_one();
            }
            n = cvar.wait(n).unwrap();
        }
    });
    {
        let (lock, cvar) = &*pair;
        let mut n = lock.lock().unwrap();
        while *n < 2000 {
            if *n % 2 == 0 {
                *n += 1;
                cvar.notify_one();
            }
            let (guard, timeout) = cvar.wait_timeout(n, Duration::from_secs(5)).unwrap();
            n = guard;
            if timeout.timed_out() {
                log!("rust_stdcheck: condvar ping-pong timed out at {}", *n);
                break;
            }
        }
        cvar.notify_all();
    }
    pong.join().ok();
    log!("rust_stdcheck: 1000 condvar round trips in {:?}", start.elapsed());
    checks.check("rust_stdcheck: Condvar round trips", *pair.0.lock().unwrap() >= 2000);

    let (tx, rx) = mpsc::channel();
    for n in 0..3u32 {
        let tx = tx.clone();
        std::thread::spawn(move || {
            for i in 0..1000 {
                tx.send(n * 1000 + i).unwrap();
            }
        });
    }
    drop(tx);
    let mut got: Vec<u32> = rx.iter().collect();
    got.sort_unstable();
    checks.check("rust_stdcheck: mpsc channel from three threads", got == (0..3000).collect::<Vec<_>>());
}

fn time(checks: &mut Checks) {
    let start = Instant::now();
    std::thread::sleep(Duration::from_millis(100));
    let slept = start.elapsed();
    log!("rust_stdcheck: sleep(100 ms) took {:?}", slept);
    checks.check("rust_stdcheck: Instant and sleep agree", slept >= Duration::from_millis(100) && slept < Duration::from_millis(150));
    let now = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH);
    log!("rust_stdcheck: SystemTime is {:?} after 1970", now.as_ref().map(|d| d.as_secs()));
    // After 2020, before 2100.
    checks.check("rust_stdcheck: SystemTime is a plausible date", now.is_ok_and(|d| (1_577_836_800..4_102_444_800).contains(&d.as_secs())));
}

fn heap(checks: &mut Checks) {
    // 64 MiB, touched: the engine's level data runs to hundreds.
    let start = Instant::now();
    let mut big = vec![0u32; 16 << 20];
    for (i, v) in big.iter_mut().enumerate().step_by(1024) {
        *v = i as u32;
    }
    let ok = big.iter().enumerate().step_by(1024).all(|(i, &v)| v == i as u32);
    log!("rust_stdcheck: 64 MiB allocated and touched in {:?}", start.elapsed());
    drop(big);
    checks.check("rust_stdcheck: a 64 MiB Vec", ok);
    let aligned = std::alloc::Layout::from_size_align(4096, 4096).unwrap();
    let p = unsafe { std::alloc::alloc(aligned) };
    checks.check("rust_stdcheck: a 4096-aligned allocation", !p.is_null() && p as usize % 4096 == 0);
    if !p.is_null() {
        unsafe { std::alloc::dealloc(p, aligned) };
    }
    let mut map = HashMap::new();
    for i in 0..10_000u32 {
        map.insert(format!("key{i}"), i);
    }
    checks.check("rust_stdcheck: HashMap with random keys", map.len() == 10_000 && map["key1234"] == 1234);
}

fn files(checks: &mut Checks) -> std::io::Result<()> {
    use std::io::{Read, Seek, SeekFrom, Write};
    let dir = std::path::Path::new(DIR);
    if dir.exists() {
        std::fs::remove_dir_all(dir)?;
    }
    std::fs::create_dir_all(dir.join("sub"))?;
    let path = dir.join(r"sub\data.bin");
    let data: Vec<u8> = (0..200_000u32).map(|i| (i * 7 % 251) as u8).collect();
    std::fs::File::create(&path)?.write_all(&data)?;
    let back = std::fs::read(&path)?;
    checks.check("rust_stdcheck: a 200 KB file written and read back", back == data);
    let mut f = std::fs::File::open(&path)?;
    f.seek(SeekFrom::Start(150_000))?;
    let mut four = [0u8; 4];
    f.read_exact(&mut four)?;
    checks.check("rust_stdcheck: seek and read", four[..] == data[150_000..150_004]);
    let len = std::fs::metadata(&path)?.len();
    checks.check("rust_stdcheck: metadata length", len == data.len() as u64);
    std::fs::write(dir.join("hello.txt"), "hello from std\n")?;
    let mut names: Vec<String> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
        .collect();
    names.sort();
    log!("rust_stdcheck: {} holds {:?}", DIR, names);
    checks.check("rust_stdcheck: read_dir lists what was made", names == ["hello.txt", "sub"]);
    // The game's data will be read like this: its own files, beside the title.
    let xex = std::fs::metadata(r"game:\rust_stdcheck.xex").map(|m| m.len());
    log!("rust_stdcheck: game:\\rust_stdcheck.xex is {:?} bytes", xex);
    checks.check("rust_stdcheck: the title's own .xex through game:", xex.is_ok_and(|n| n > 0));
    Ok(())
}

#[unsafe(no_mangle)]
pub extern "C" fn main(_argc: i32, _argv: *const *const u8) -> i32 {
    std::panic::set_hook(Box::new(|info| {
        log!("panic: {}", info);
        unsafe { xsdk_rt::exit(3) };
    }));
    let mut checks = Checks::default();
    let start = Instant::now();
    log!("rust_stdcheck: Rust's std on xenon-forge's backend in an xsdk title; it ends by itself");
    threads(&mut checks);
    sync(&mut checks);
    time(&mut checks);
    heap(&mut checks);
    if let Err(e) = files(&mut checks) {
        log!("rust_stdcheck: file error: {} ({:?})", e, e.kind());
        checks.check("rust_stdcheck: files", false);
    }
    log!("rust_stdcheck: done in {:?}", start.elapsed());
    checks.finish()
}
