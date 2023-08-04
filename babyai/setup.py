from setuptools import setup

setup(
    name='babyai',
    version='0.0.2',
    license='BSD 3-clause',
    keywords='memory, environment, agent, rl, openaigym, openai-gym, gym',
    packages=['babyai', 'babyai.levels', 'babyai.utils'],
    install_requires=[
        'gym',
        'numpy', # Temporary: fix numpy version because of bug introduced in 1.16
        'pyqt5',
        "torch",
        'blosc',
        'minigrid'
    ],
)
