from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="angry_pixie_pricing",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[req for req in requirements if not req.startswith('#') and req.strip()],
    entry_points={
        'console_scripts': [
            'angry-pixie=angry_pixie_pricing.main:cli',
        ],
    },
    author="Your Name",
    description="European Electricity Price Analysis Tool",
    python_requires=">=3.8",
)