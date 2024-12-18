# Based on https://answers.ros.org/question/397319/how-to-copy-folders-with-subfolders-to-package-installation-path/

import os
from setuptools import find_packages, setup

package_name = 'week_6'

data_files=[
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
]

def package_files(data_files, directory_list):

    paths_dict = {}

    for directory in directory_list:

        for (path, directories, filenames) in os.walk(directory):

            for filename in filenames:

                file_path = os.path.join(path, filename)
                install_path = os.path.join('share', package_name, path)

                if install_path in paths_dict.keys():
                    paths_dict[install_path].append(file_path)
                else:
                    paths_dict[install_path] = [file_path]

    for key in paths_dict.keys():
        data_files.append((key, paths_dict[key]))

    return data_files


setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=package_files(data_files, ['models/', 'launch/', 'worlds/', 'rviz/']),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pfr500',
    maintainer_email='pedro.ribeiro@york.ac.uk',
    description='Week 6 practical',
    license='EPL-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lidar_fault = week_6.lidar_fault:main',
            'robot_controller = week_6.robot_controller:main',
        ],
    },
)
