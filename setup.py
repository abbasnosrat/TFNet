from setuptools import setup

setup(
    name='TFNet',
    version='1.0',
    install_requires=[
        "control~=0.9.2",
        "numpy~=1.21.6",
        "pandas~=1.4.4",
        "torch~=1.12.0+cu116",
        "tqdm~=4.64.1"],
    packages=['TFNet'],
    include_package_data=True,
    url='#',
    license='MIT',
    author='Abbas Nosrat',
    author_email='abbasnosrat@gmail.com',
    description='A few-shot system identification method for lti systems based on CNNs'
)
