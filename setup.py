"""Setup script for RENTT package."""

from pathlib import Path

from setuptools import find_packages, setup

# Read README

readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

def read_requirements(filename):
    req_file = Path(__file__).parent / filename
    if req_file.exists():
        with open(req_file, encoding="utf-8") as f:
            return [
                line.strip()
                for line in f
                if line.strip()
                and not line.startswith("#")
                and not line.startswith("-r")
            ]
    return []

setup(
    name="RENTT",
    version="0.1.0",
    author="Helena Monke, Yilin Chen",
    author_email="helena.monke@ipa.fraunhofer.de",
    description="Runtime Efficient Network to Tree Transformation (RENTT)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=read_requirements("requirements.txt"),
    python_requires=">=3.10, <3.13",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)