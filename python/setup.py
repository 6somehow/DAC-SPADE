import os
import shutil

from setuptools import find_packages, setup

CURRENT_DIR = os.path.dirname(__file__)
libinfo_py = os.path.join(CURRENT_DIR, "spade", "_libinfo.py")
libinfo = {}
with open(libinfo_py, "r") as f:
    exec(f.read(), libinfo)
__version__ = libinfo["__version__"]


def gen_file_list(srcs, f_cond):
    file_list = []
    for src in srcs:
        for root, _, files in os.walk(src):
            value = []
            for file in files:
                if f_cond(file):
                    path = os.path.join(root, file)
                    value.append(path.replace("spade/", ""))
            file_list.extend(value)
    return file_list


def gen_autotune_common_file_list():
    srcs = ["spade/autotune"]
    f_cond = lambda x: True if x.endswith(".py") else False
    return gen_file_list(srcs, f_cond)


def gen_utils_file_list():
    srcs = ["spade/utils"]
    f_cond = lambda x: True if x.endswith(".py") else False
    return gen_file_list(srcs, f_cond)


def gen_engine_file_list():
    srcs = ["spade/engine"]
    f_cond = lambda x: True if x.endswith(".py") else False
    return gen_file_list(srcs, f_cond)


def gen_core_file_list():
    srcs = ["spade/core"]
    f_cond = lambda x: True if x.endswith(".py") else False
    return gen_file_list(srcs, f_cond)


setup_kwargs = {}
include_libs = True
wheel_include_libs = True

setup(name="spade",
      version=__version__,
      description="Spade: Sparse DiT engine",
      zip_safe=True,
      install_requires=["triton"],
      packages=find_packages(),
      package_data={
          "spade":
          [] + gen_utils_file_list() + gen_autotune_common_file_list() +
          gen_engine_file_list() + gen_core_file_list()
      },
      python_requires=">=3.7, <4",
      **setup_kwargs)
