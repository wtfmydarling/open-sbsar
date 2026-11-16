//! open-sbsar: a clean-room engine for compiled Substance archives (`.sbsar`).
//!
//! Open a package, inspect a graph's inputs and outputs, set parameters and render the outputs
//! on the GPU (feature `gpu`, Vulkan through wgpu) or on the CPU reference path. See [`Package`],
//! [`Parameters`] and [`Renderer`]; the format itself is specified in the repository's `docs/`.

mod api;
pub use api::{Error, Graph, InputImage, Instance, OutputImage, Package, Parameters, PixelFormat, RenderStats, Renderer, Result, Value};
pub use package::{InputInfo, InputKind, OutputInfo};
/// The wgpu version this crate uses (for [`Renderer::with_device`]).
#[cfg(feature = "gpu")]
pub use wgpu;

// Engine internals, public for the command-line tool and research probes; not a stable API.
#[doc(hidden)]
pub mod envc;
#[doc(hidden)]
pub mod fxmap;
#[cfg(feature = "gpu")]
#[doc(hidden)]
pub mod gpu;
#[cfg(feature = "gpu")]
#[doc(hidden)]
pub mod gpu_pp;
#[doc(hidden)]
pub mod image;
mod jpeg;
#[doc(hidden)]
pub mod ops;
#[doc(hidden)]
pub mod package;
#[doc(hidden)]
pub mod program;
#[doc(hidden)]
pub mod sevenz;
mod text;
#[doc(hidden)]
pub mod vm;

/// A safety limit against malformed programs was reached (real files never reach them;
/// SDRT_CAPLOG=1 reports it on stderr).
#[doc(hidden)]
pub fn cap_hit(what: &str) {
    if envc::var("SDRT_CAPLOG").is_ok() {
        eprintln!("SDRT_CAP {what}");
    }
}

/// Research aid (SDRT_CAPLOG=1): report new maxima of VM expression nesting above 48.
#[doc(hidden)]
pub fn depth_seen(d: usize) {
    static MAX: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(48);
    if MAX.fetch_max(d, std::sync::atomic::Ordering::Relaxed) < d && envc::var("SDRT_CAPLOG").is_ok() {
        eprintln!("SDRT_DEPTH {d}");
    }
}
