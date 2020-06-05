from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='carla_rllib',
      version='1.0',
      description='Reinforcement Library for the CARLA Simulator',
      classifiers=[
          'License :: MIT License',
          'Programming Language :: Python :: 2.7/3.6',
          'Topic :: Reinforcement Learning',
      ],
      url='https://github.com/50sven/carla_rllib',
      author='Sven Müller',
      author_email='sven.mueller92@gmx.de',
      license='MIT',
      packages=['carla_rllib'],
      install_requires=requirements,
      zip_safe=False
      )
