import os
import importlib.util
import hashlib


class TITCompiler:
    """
    A compiler for Triton kernels.
    It handles caching and dynamic module loading from a code string.
    """

    def __init__(self, dir_path="workspace/triton_build"):
        self.dir_path = dir_path
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)

    def get_code_hash(self, code: str):
        """Computes the SHA256 hash of a string."""
        return hashlib.sha256(code.encode()).hexdigest()

    def compile_module(self, kernel_code: str, kernel_name: str):
        """
        Compiles the given Triton kernel code string into a Python module.
        """
        # Create a unique filename based on the hash of the code
        code_hash = self.get_code_hash(kernel_code)
        tmp_filename = os.path.join(self.dir_path,
                                    f"{kernel_name}_{code_hash}.py")

        # If the file doesn't exist, write it so inspect can find the source
        if not os.path.exists(tmp_filename):
            with open(tmp_filename, 'w') as f:
                f.write(kernel_code)

        # Import the file as a module
        spec = importlib.util.spec_from_file_location(kernel_name,
                                                      tmp_filename)
        kernel_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kernel_module)
        return kernel_module

    def compile_kernel(self, kernel_code: str, kernel_name: str):
        kernel_module = self.compile_module(kernel_code, kernel_name)
        return getattr(kernel_module, kernel_name)
