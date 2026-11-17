from glob import glob

from setuptools import find_packages, setup

package_name = "peg_trial"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["tests"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Team",
    maintainer_email="you@example.com",
    description="Tri-Wizard Task 1: peg sorting",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "vision_node = peg_trial.nodes.vision_node:main",
            "autonomy_node = peg_trial.nodes.autonomy_node:main",
            "teleop_keyboard = peg_trial.control.teleop_keyboard:main",
        ],
    },
)
