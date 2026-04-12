import os
import sys  # <-- Import sys module
import torch
import hashlib
import shutil
import importlib.util
from torch.utils.cpp_extension import load


class CUDATCompiler:
    """
    A compiler for CUDA kernels using PyTorch's Just-In-Time (JIT) compilation,
    with an explicit caching mechanism to save and load compiled .so files.

    This class compiles CUDA source code into a shared library (.so file) and
    caches it in a specified directory. On subsequent runs, it loads the cached
    .so file directly, avoiding the need for recompilation.
    """

    def __init__(self, cache_dir="workspace/cuda_cache"):
        """
        Initializes the compiler and creates the cache and build directories.

        Args:
            cache_dir (str): The directory to store and load compiled .so files.
                             Build artifacts will be stored in a 'build' subdirectory.
        """
        self.cache_dir = cache_dir
        # A subdirectory for temporary build files from torch.utils.cpp_extension
        self.build_dir = os.path.join(self.cache_dir, "build")
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.build_dir, exist_ok=True)
        # In-memory cache for modules loaded during the current session
        self.module_table = {}

    def _ensure_cuda_arch_list(self):
        """
        Ensure JIT compilation targets the current GPU architecture when the user
        did not specify TORCH_CUDA_ARCH_LIST explicitly.
        """
        if os.environ.get("TORCH_CUDA_ARCH_LIST"):
            return
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

    def _get_build_signature(self) -> str:
        """
        Build a stable signature for binary compatibility. This avoids loading a
        cached .so compiled on a different GPU arch or CUDA/PyTorch runtime.
        """
        major, minor = torch.cuda.get_device_capability()
        arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
        cuda_ver = torch.version.cuda or ""
        torch_ver = torch.__version__
        return f"cc{major}{minor}_arch{arch_list}_cuda{cuda_ver}_torch{torch_ver}"

    def get_code_hash(self, code: str) -> str:
        """
        Computes the SHA256 hash of a source code string to generate a unique identifier.

        Args:
            code (str): The source code to hash.

        Returns:
            str: The hexadecimal hash digest.
        """
        return hashlib.sha256(code.encode()).hexdigest()

    def compile_module(self,
                       kernel_code: str,
                       module_name: str,
                       verbose: bool = False):
        """
        Compiles CUDA C++ code into a loadable module, using a cache for .so files.

        The method first checks an in-memory cache. If not found, it checks for a
        pre-compiled .so file on disk. If that also isn't found, it compiles the
        kernel, saves the resulting .so file to the cache directory, and then loads it.

        Args:
            kernel_code (str): The CUDA source code to compile.
            module_name (str): A base name for the compiled module, used for caching.
            verbose (bool): If True, enables verbose output from the compiler.

        Returns:
            A loaded Python module containing the compiled CUDA functions.
        """
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available, cannot compile CUDA kernel.")

        self._ensure_cuda_arch_list()
        build_sig = self._get_build_signature()
        code_hash = self.get_code_hash(f"{build_sig}\n{kernel_code}")
        # This is the actual, unique module name that PyTorch bakes into the .so file.
        # It MUST be used for both compilation and loading from cache.
        jit_module_name = f"{module_name}_{code_hash}"

        # 1. Check in-memory cache (fastest)
        if code_hash in self.module_table:
            return self.module_table[code_hash]

        # 2. Check on-disk cache for a pre-compiled .so file
        so_filename = f"{module_name}_{code_hash[:10]}.so"
        so_path = os.path.join(self.cache_dir, so_filename)

        if os.path.exists(so_path):
            try:
                # When loading from cache, use the unique JIT module name.
                spec = importlib.util.spec_from_file_location(
                    jit_module_name, so_path)

                # Guard against invalid or corrupted .so files
                if spec is None:
                    raise ImportError(
                        f"Could not load spec for module '{jit_module_name}' from {so_path}"
                    )

                loaded_module = importlib.util.module_from_spec(spec)
                sys.modules[
                    jit_module_name] = loaded_module  # Add to sys.modules
                spec.loader.exec_module(loaded_module)

                self.module_table[code_hash] = loaded_module
                return loaded_module
            except ImportError as e:
                print(
                    f"Warning: Failed to load cached module {so_path}. Recompiling. Error: {e}"
                )

        # 3. If no valid cache exists, compile the kernel.
        source_filename = f"{jit_module_name}.cu"
        source_path = os.path.join(self.build_dir, source_filename)
        with open(source_path, 'w') as f:
            f.write(kernel_code)

        # Use the unique JIT module name for compilation.
        compiled_module = load(name=jit_module_name,
                               sources=[source_path],
                               build_directory=self.build_dir,
                               verbose=verbose)

        # 4. Save the new .so file to our permanent cache directory.
        shutil.copy(compiled_module.__file__, so_path)

        # 5. Add to in-memory cache and return.
        self.module_table[code_hash] = compiled_module

        return compiled_module
