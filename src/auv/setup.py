from setuptools import find_packages, setup

package_name = 'auv'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',  # Required for setuptools to work
        'torch',        # PyTorch (required for YOLOv5)
        'opencv-python', # OpenCV (for image processing)
        'numpy',         # NumPy is commonly used for numerical operations
        'rclpy',         # ROS 2 Python client library
        'std_msgs',      # Standard ROS 2 messages
    ],
    zip_safe=True,
    maintainer='agastya',
    maintainer_email='agastya@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'flareNode = auv.orange_flare:main',
            'gateNode = auv.gate_right:main',
            'tubFrontNode = auv.blue_front_left:main',
            'dropNode = auv.drop:main',
            'gripNode = auv.grip:main',
            'pingNode = auv.ping:main',
            'RYBNode1 = auv.ryb_flare_1:main',
            'RYBNode2 = auv.ryb_flare_2:main',
            'RYBNode3 = auv.ryb_flare_3:main',
            'mainFinalNode = auv.main_final:main',
            'mainGDGNode = auv.main_gate_drop_grip:main',
            'mainGFNode = auv.main_flare_gate:main',
        ],
    },
)
