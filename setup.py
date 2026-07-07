# Credit to: https://github.com/machines-in-motion/mim_robots.git

from os import path, walk
from setuptools import setup, find_packages


def find_resources(package_name):
    """ Find the relative path of files under the resource folder. """
    resources = []
    package_dir = path.join(package_name)
    resources_dir = package_dir 

    for (root, _, files) in walk(resources_dir):
        for afile in files:
            if (
                afile != package_name
                and not afile.endswith(".DS_Store")
                and not afile.endswith(".py")
            ):
                rel_dir = path.relpath(root, package_dir)
                src = path.join(rel_dir, afile)
                resources.append(src)
    return resources


# Package name.
package_name = "KinovaPy"

# Find the resource files.
resources = find_resources(package_name)

# Final setup.
setup(
    name=package_name,
    version="0.1.0",
    package_dir={package_name: path.join(package_name)},
    packages=[package_name] + [package_name + '.' + pkg for pkg in find_packages(path.join(package_name))], 
    package_data={package_name: resources},
    install_requires=[
    ],
    python_requires=">=3.8",
)