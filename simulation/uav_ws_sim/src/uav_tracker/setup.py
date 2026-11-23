from setuptools import setup

package_name = 'uav_tracker'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ling',
    maintainer_email='ling@localhost',
    description='UAV autonomous mission control node',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'uav_tracker = uav_tracker.uav_tracker:main',
        ],
    },
)
