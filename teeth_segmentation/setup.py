from setuptools import setup, find_packages

setup(
    name='mmdet_custom',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[],  # 의존성 제거
    zip_safe=False,
)
