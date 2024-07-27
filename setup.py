import setuptools

setuptools.setup(
    name="sbe",
    version="0.4.1",
    author="Chris Yuen",
    author_email="chris@kizzx2.com",
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/kizzx2/sbe-python",
    install_requires=["lxml", "bitstring"],
    packages=setuptools.find_packages(exclude=["tests"]),
    include_package_data=True,
)
