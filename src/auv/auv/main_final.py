
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

def pub6File():
    return LaunchDescription([
        Node(
            package='auv',     
            executable='RYBNode1',    
            name='ryb_node_1',     
            output='screen',           
            parameters=[{'topic': 'command'}] 
        )
    ])

def pub7File():
    return LaunchDescription([
        Node(
            package='auv',     
            executable='RYBNode2',    
            name='ryb_node_2',     
            output='screen',           
            parameters=[{'topic': 'command'}] 
        )
    ])

def pub8File():
    return LaunchDescription([
        Node(
            package='auv',     
            executable='RYBNode3',    
            name='ryb_node_3',     
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

    launch_service_3 = LaunchService()
    launch_service_3.include_launch_description(pub3File())
    launch_service_3.run()  # Blocks until the first node terminates

    launch_service_4 = LaunchService()
    launch_service_4.include_launch_description(pub4File())
    launch_service_4.run() 

    launch_service_5 = LaunchService()
    launch_service_5.include_launch_description(pub5File())
    launch_service_5.run() 

    launch_service_6 = LaunchService()
    launch_service_6.include_launch_description(pub6File())
    launch_service_6.run() 

    launch_service_7 = LaunchService()
    launch_service_7.include_launch_description(pub7File())
    launch_service_7.run() 

    launch_service_8 = LaunchService()
    launch_service_8.include_launch_description(pub8File())
    launch_service_8.run() 

    print("All nodes finished. micro_ros_agent still running in the background.")

if __name__ == '__main__':
    main()
