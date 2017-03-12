import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):

    # user_options are best specified in setup.cfg, this works for both pip and 'python setup.py'.
    # Via command-line, it might be possible like this:
    #     setup.py: $ python setup.py --help build_ext
    #     For pip, it might be something like that but it doesn't work:
    #         $ pip install .\eos --build-option="--opencv-dir=C:/opencv/install"
    # So just use setup.cfg instead.
    # -----
    # The second argument is an optional short option (can e.g. be 'b').
    # It seems options need to be lower-case.
    user_options = build_ext.user_options + [
        ('boost-root', None, "Boost search location"),
        ('eigen3-include-dir', None, "Directory of the Eigen3 headers"),
        ('opencv-dir', None, "OpenCV config dir, where OpenCVConfig.cmake can be found")
    ]

    def initialize_options(self):
        build_ext.initialize_options(self)
        self.boost_root = None
        self.eigen3_include_dir = None
        self.opencv_dir = None

    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable,
                      '-DEOS_BUILD_EXAMPLES=OFF',
                      '-DEOS_GENERATE_PYTHON_BINDINGS=ON'
                      ]
        if self.opencv_dir:
            cmake_args += ['-DOpenCV_DIR:PATH=' + self.opencv_dir]
        if self.boost_root:
            cmake_args += ['-DBOOST_ROOT:PATH=' + self.boost_root]
        if self.eigen3_include_dir:
            cmake_args += ['-DEIGEN3_INCLUDE_DIR:PATH=' + self.eigen3_include_dir]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-G', 'Visual Studio 14 2015 Win64']
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

setup(
    name='eos-py',
    version='0.12.1',
    author='Patrik Huber',
    author_email='patrikhuber@gmail.com',
    description='Python bindings for eos - A lightweight 3D Morphable Face Model fitting library in modern C++11/14',
    long_description='',
    url='https://github.com/patrikhuber/eos',
    license='Apache-2.0',
    ext_modules=[CMakeExtension('eos')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
