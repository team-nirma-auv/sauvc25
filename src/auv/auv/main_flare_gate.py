
from launch import LaunchService
from launch import LaunchDescription
from launch_ros.actions import Node
import time

def pub1File():
    return LaunchDescription([
        Node(
            package='auv',     
            executable='flareNode',    
            name='flare_node',     
            output='screen',           
            parameters=[{'topic': 'command'}] 
        )
    ])

def pub2File():
    return LaunchDescription([
        Node(
            package='auv',          
            executable='gateNode',      
            name='gate_node',    
            output='screen',             
            parameters=[{'topic': 'command'}] 
        )
    ])

def main():
    time.sleep(10)
    
    launch_service_1 = LaunchService()
    launch_service_1.include_launch_description(pub1File())
    launch_service_1.run()  # Blocks until the first node terminates
    
    launch_service_2 = LaunchService()
    launch_service_2.include_launch_description(pub2File())
    launch_service_2.run()  # Blocks until the first node terminates

    print("All nodes finished. micro_ros_agent still running in the background.")

if __name__ == '__main__':
    main()
