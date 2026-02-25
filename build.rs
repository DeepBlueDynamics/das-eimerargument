// build.rs — Compile CUDA kernels to PTX via nvcc
//
// Only runs when the "gpu" feature is enabled.
// Detects CUDA Toolkit via CUDA_PATH env var or standard install locations.
// Compiled PTX is written to OUT_DIR for include_bytes! in src/gpu/mod.rs.

fn main() {
    #[cfg(feature = "gpu")]
    gpu_build();
}

#[cfg(feature = "gpu")]
fn gpu_build() {
    use std::env;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernel_dir = Path::new("kernels");

    // Kernel .cu files to compile
    let kernels = [
        "weber_kernel.cu",
        "bond_kernel.cu",
        "integrate_kernel.cu",
        "field_residual_kernel.cu",
    ];

    // Rerun triggers
    for kernel in &kernels {
        println!("cargo:rerun-if-changed=kernels/{}", kernel);
    }
    println!("cargo:rerun-if-changed=kernels/dd_math.cuh");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=ProgramFiles");

    // Try to find CUDA toolkit
    let cuda_path = match find_cuda_path() {
        Some(p) => p,
        None => {
            // CUDA not found — create stub PTX files so the Rust code compiles.
            // At runtime, cudarc will fail when trying to init CUDA, which is
            // caught by the GPU fallback logic in the binaries.
            println!("cargo:warning=CUDA Toolkit not found. Creating stub PTX files.");
            println!("cargo:warning=Install CUDA Toolkit 12.x and set CUDA_PATH to build real GPU kernels.");

            for kernel in &kernels {
                let ptx_name = kernel.replace(".cu", ".ptx");
                let ptx_path = out_dir.join(&ptx_name);
                std::fs::write(&ptx_path, "// STUB — CUDA Toolkit not available at build time\n")
                    .unwrap_or_else(|e| panic!("Cannot write stub {}: {}", ptx_name, e));
            }
            return;
        }
    };

    let nvcc = if cuda_path.join("bin").join("nvcc.exe").exists() {
        cuda_path.join("bin").join("nvcc.exe")
    } else if cuda_path.join("bin").join("nvcc").exists() {
        cuda_path.join("bin").join("nvcc")
    } else {
        // CUDA_PATH exists but nvcc not found — partial install. Fall back to stubs.
        println!("cargo:warning=CUDA_PATH set to {:?} but nvcc not found in bin/.", cuda_path);
        println!("cargo:warning=Re-run CUDA Toolkit installer with Custom > Development > Compiler checked.");
        println!("cargo:warning=Creating stub PTX files for now.");

        for kernel in &kernels {
            let ptx_name = kernel.replace(".cu", ".ptx");
            let ptx_path = out_dir.join(&ptx_name);
            std::fs::write(&ptx_path, "// STUB — nvcc not found in CUDA_PATH\n")
                .unwrap_or_else(|e| panic!("Cannot write stub {}: {}", ptx_name, e));
        }
        return;
    };

    let nvcc_str = nvcc.to_str().unwrap();

    // Find MSVC cl.exe for nvcc host compiler
    let ccbin = find_msvc_cl();
    if let Some(ref cl_dir) = ccbin {
        println!("cargo:warning=Found MSVC cl.exe at: {:?}", cl_dir);
    } else {
        println!("cargo:warning=MSVC cl.exe not found — nvcc may fail. Install Visual Studio Build Tools.");
    }

    for kernel in &kernels {
        let cu_path = kernel_dir.join(kernel);
        let ptx_name = kernel.replace(".cu", ".ptx");
        let ptx_path = out_dir.join(&ptx_name);

        if !cu_path.exists() {
            panic!("Kernel source not found: {:?}", cu_path);
        }

        let mut cmd = Command::new(nvcc_str);
        cmd.args(&[
            "--ptx",
            "-arch=sm_86",          // RTX 3060
            "-O3",
            "--use_fast_math",       // OK for f64 FMA path (doesn't affect dd correctness)
            "-o",
            ptx_path.to_str().unwrap(),
            "-I",
            kernel_dir.to_str().unwrap(),
            cu_path.to_str().unwrap(),
        ]);

        // Point nvcc at MSVC host compiler
        if let Some(ref cl_dir) = ccbin {
            cmd.arg("-ccbin");
            cmd.arg(cl_dir.to_str().unwrap());
        }

        let status = cmd
            .status()
            .unwrap_or_else(|e| panic!("Failed to run nvcc: {}", e));

        if !status.success() {
            panic!("nvcc failed to compile {}", kernel);
        }

        // Patch PTX ISA version for driver compatibility.
        // nvcc 12.9 generates .version 8.8 but CUDA driver 12.6 only supports up to 8.5.
        // Our kernels only use basic f64/dd arithmetic — no 8.8-specific instructions.
        patch_ptx_version(&ptx_path);

        println!("cargo:warning=Compiled {} -> {}", kernel, ptx_name);
    }
}

#[cfg(feature = "gpu")]
fn has_nvcc(p: &std::path::Path) -> bool {
    p.join("bin").join("nvcc.exe").exists() || p.join("bin").join("nvcc").exists()
}

#[cfg(feature = "gpu")]
fn find_cuda_path() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;

    // 1. CUDA_PATH environment variable (only if nvcc is actually there)
    if let Ok(path) = std::env::var("CUDA_PATH") {
        let p = PathBuf::from(&path);
        if has_nvcc(&p) {
            return Some(p);
        }
    }

    // 2. Standard Windows locations
    let program_files = std::env::var("ProgramFiles")
        .unwrap_or_else(|_| "C:\\Program Files".to_string());
    let cuda_base = PathBuf::from(&program_files).join("NVIDIA GPU Computing Toolkit").join("CUDA");

    if cuda_base.exists() {
        // Find highest version
        if let Ok(entries) = std::fs::read_dir(&cuda_base) {
            let mut versions: Vec<PathBuf> = entries
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.is_dir())
                .collect();
            versions.sort();
            if let Some(latest) = versions.last() {
                if has_nvcc(latest) {
                    return Some(latest.clone());
                }
            }
        }
    }

    // 3. Conda environment (nvcc installed via conda)
    if let Ok(conda_prefix) = std::env::var("CONDA_PREFIX") {
        let conda_lib = PathBuf::from(&conda_prefix).join("Library");
        if conda_lib.join("bin").join("nvcc.exe").exists() {
            return Some(conda_lib);
        }
    }
    // Also check common conda paths directly
    if let Ok(userprofile) = std::env::var("USERPROFILE") {
        for conda_dir in &["miniconda3", "anaconda3", "miniforge3"] {
            let conda_lib = PathBuf::from(&userprofile).join(conda_dir).join("Library");
            if conda_lib.join("bin").join("nvcc.exe").exists() {
                return Some(conda_lib);
            }
        }
    }

    // 4. Linux standard
    let linux_default = PathBuf::from("/usr/local/cuda");
    if linux_default.exists() {
        return Some(linux_default);
    }

    None
}

/// Patch PTX ISA version for driver compatibility.
/// nvcc 12.9 generates `.version 8.8` but the installed CUDA driver (12.6) only
/// supports up to PTX ISA 8.5. Our kernels use only basic f64/dd arithmetic and
/// shared memory — no features beyond ISA 7.0 — so downgrading is safe.
#[cfg(feature = "gpu")]
fn patch_ptx_version(ptx_path: &std::path::Path) {
    let content = std::fs::read_to_string(ptx_path)
        .unwrap_or_else(|e| panic!("Cannot read PTX {:?}: {}", ptx_path, e));

    // Match `.version X.Y` where X.Y > 8.5
    if let Some(pos) = content.find(".version ") {
        let after = &content[pos + 9..];
        if let Some(newline) = after.find('\n') {
            let version_str = after[..newline].trim();
            // Parse major.minor
            let parts: Vec<&str> = version_str.split('.').collect();
            if parts.len() == 2 {
                if let (Ok(major), Ok(minor)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>()) {
                    if major > 8 || (major == 8 && minor > 5) {
                        let old = format!(".version {}", version_str);
                        let new = ".version 8.5";
                        let patched = content.replacen(&old, new, 1);
                        std::fs::write(ptx_path, patched)
                            .unwrap_or_else(|e| panic!("Cannot write patched PTX: {}", e));
                        println!("cargo:warning=Patched PTX version {} -> 8.5 for driver compat",
                                 version_str);
                    }
                }
            }
        }
    }
}

/// Find the directory containing MSVC cl.exe for nvcc's -ccbin flag.
/// Searches VS2022, VS2019, VS2017, and VS Build Tools in both Program Files locations.
#[cfg(feature = "gpu")]
fn find_msvc_cl() -> Option<std::path::PathBuf> {
    use std::path::PathBuf;

    let program_dirs = [
        std::env::var("ProgramFiles(x86)")
            .unwrap_or_else(|_| "C:\\Program Files (x86)".to_string()),
        std::env::var("ProgramFiles")
            .unwrap_or_else(|_| "C:\\Program Files".to_string()),
    ];

    let editions = [
        "Enterprise", "Professional", "Community", "BuildTools",
    ];
    let years = ["2022", "2019", "2017"];

    for pf in &program_dirs {
        for year in &years {
            for edition in &editions {
                let vs_dir = PathBuf::from(pf)
                    .join("Microsoft Visual Studio")
                    .join(year)
                    .join(edition)
                    .join("VC")
                    .join("Tools")
                    .join("MSVC");

                if !vs_dir.exists() {
                    continue;
                }

                // Find highest MSVC version
                if let Ok(entries) = std::fs::read_dir(&vs_dir) {
                    let mut versions: Vec<PathBuf> = entries
                        .filter_map(|e| e.ok())
                        .map(|e| e.path())
                        .filter(|p| p.is_dir())
                        .collect();
                    versions.sort();

                    if let Some(latest) = versions.last() {
                        let cl_dir = latest.join("bin").join("Hostx64").join("x64");
                        if cl_dir.join("cl.exe").exists() {
                            return Some(cl_dir);
                        }
                    }
                }
            }
        }
    }

    None
}
