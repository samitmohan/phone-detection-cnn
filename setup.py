from setuptools import setup, find_packages

setup(
    name='mlp-phone-detector',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
        'torch',
        'mmpose',
        'scikit-learn', 
        'tqdm', 
    ],
    entry_points={
        'console_scripts': [
            'detect-phone-mlp=mlp_detector.cli:main',
        ],
    },
    author='Samit', 
    description='A keypoint-based phone detection system using MLP and HRNet pose estimation.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/samitmohan/phone-detection-cnn/', 
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)