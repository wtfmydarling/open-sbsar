//! open-sbsar: a clean-room engine for compiled Substance archives (`.sbsar`).
//!
//! Open a package, inspect a graph's inputs and outputs, set parameters and render the outputs
//! on the GPU (feature `gpu`, Vulkan through wgpu) or on the CPU reference path. See [`Package`],
//! [`Parameters`] and [`Renderer`]; the format itself is specified in the repository's `docs/`.

mod api;
pub use api::{Error, Graph, InputImage, OutputImage, Package, Parameters, PixelFormat, Renderer, Result, Value};
pub use package::{InputInfo, InputKind, OutputInfo};

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
mod text;
#[doc(hidden)]
pub mod vm;
