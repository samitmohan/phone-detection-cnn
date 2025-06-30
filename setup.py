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
        'scikit-learn', # For metrics if needed, though not directly in detector.py
        'tqdm', # For data_collector, but good to include for full project
    ],
    entry_points={
        'console_scripts': [
            'detect-phone-mlp=mlp_detector.cli:main',
        ],
    },
    author='Your Name', # Replace with your name
    author_email='your.email@example.com', # Replace with your email
    description='A keypoint-based phone detection system using MLP and HRNet pose estimation.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/your_github_username/mlp-phone-detector', # Replace with your project's GitHub URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)