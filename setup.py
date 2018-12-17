import os
import glob
import sys
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import setuptools

__version__ = '0.1.0'

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

ccfiles =  glob.glob('*.cc') + glob.glob('src/*.cc')
print("source files:", ccfiles)
ext_modules=[
    Extension(
        'echainer_internal',
        ccfiles,
        language='c++',
        ## Somehow LD_LIBRARY_PATH is not working for this....
        ## Use result of `pkg-config --libs-only-L protobuf grpc++ grpc` if not installed in /usr/lib
        library_dirs=['/home/kuenishi/local/lib'],
        include_dirs=[
            #'/home/kuenishi/local/include',
            #'/usr/local/cuda/include',
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        libraries=['glog','protobuf', 'grpc++', 'grpc', 'grpc++_reflection', 'dl', 'cppetcd'],
    ),
]

class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
    parallel = 4

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])

        assert ct == 'unix'
        # TODO: Somehow currently Python 3.6 uses gcc compiler by
        # default, not g++ nor c++, even not $CXX defined one.
        # In distutils: https://bugs.python.org/issue1222585
        # self.compiler.compiler_so.remove("-Wstrict-prototypes")
        opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
        # The c++14 is prefered over c++11 (when it is available).
        opts.append('-std=c++14')
        opts.append('-fvisibility=hidden')
        ## Sanitizers https://github.com/google/sanitizers/wiki
        ## Will need python built with sanitiers https://devguide.python.org/clang/
        #opts.append('-pie')
        # Turn on address sanitizer
        #opts.append('-fsanitize=address')
        #opts.append('-fno-omit-frame-pointer')
        # Turn on thread sanitizer
        #opts.append('-fsanitize=thread')
        # Turn on memory sanitizer
        #opts.append('-fsanitize=memory')
        #opts.append('-fsanitize-memory-track-origins')

        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = ['-g']
        build_ext.build_extensions(self)

setup(
    name='echainer',
    version=__version__,
    author="Kota UENISHI",
    author_email="kota@preferred.jp",
    url="hoge",
    description="Elasticity communicator for Chainer",
    long_description="><",
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.2', 'six', 'jinja2', 'chainer>=6.0.0b1'],
    #, 'cupy-cuda92'], # requires cupy for nccl communicator
    cmdclass={'build_ext': BuildExt},
    # test_requires=['pytest', 'pytest-xdist'],
    zip_safe=False,
    )
