//! The process environment read once: `var` is `std::env::var` over a snapshot taken at first use.
//! The test switches (SDRT_*) are looked up inside per-pixel and per-stamp code, where the real
//! lookup (a global lock, a scan, an allocation) cost seconds on large nodes. Only SDRT_* variables
//! are kept (every caller asks for one; usually none is set, so a lookup is a scan of an empty or
//! tiny list). Nothing in sdrt sets variables at run time.

use std::sync::OnceLock;

static ENV: OnceLock<Vec<(String, String)>> = OnceLock::new();

pub fn var(k: &str) -> Result<String, std::env::VarError> {
    let env = ENV.get_or_init(|| std::env::vars().filter(|(n, _)| n.starts_with("SDRT_")).collect());
    if !k.starts_with("SDRT_") {
        return std::env::var(k);
    }
    env.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone()).ok_or(std::env::VarError::NotPresent)
}

/// `var` read once per call site, for hot paths: same `Result` shape as `var` (with `&'static str`).
#[macro_export]
macro_rules! env_once {
    ($k:expr) => {{
        static C: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
        C.get_or_init(|| $crate::envc::var($k).ok()).as_deref().ok_or(std::env::VarError::NotPresent)
    }};
}
