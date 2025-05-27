
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

def pub3File():
    return LaunchDescription([
        Node(
            package='auv',     
            executable='tubFrontNode',    
            name='drop_front_node',     
            output='screen',           
            parameters=[{'topic': 'command'}] 
        )
    ])

def pub4File():
    return LaunchDescription([
        Node(
            package='auv',          
            executable='dropNode',      
            name='drop_bottom_node',    
            output='screen',             
            parameters=[{'topic': 'command'}] 
        )
    ])

def pub5File():
    return LaunchDescription([
        Node(
            package='auv',          
            executable='gripNode',      
            name='grip_bottom_node',    
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
    print("Launching first node...")
    launch_service_3 = LaunchService()
    launch_service_3.include_launch_description(pub3File())
    launch_service_3.run()  # Blocks until the first node terminates
    print("First node finished.")

    print("Launching second node...")
    launch_service_4 = LaunchService()
    launch_service_4.include_launch_description(pub4File())
    launch_service_4.run() 
    print("Second node finished.")

    print("Launching Third node...")
    launch_service_5 = LaunchService()
    launch_service_5.include_launch_description(pub5File())
    launch_service_5.run() 

    print("All nodes finished. micro_ros_agent still running in the background.")

if __name__ == '__main__':
    main()
