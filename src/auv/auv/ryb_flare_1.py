import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from std_msgs.msg import Float32MultiArray
import time

class SonarSubscriber(Node):
    def __init__(self):
        super().__init__('sonar_subscriber')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'ping_topic',
            self.listener_callback,
            10)
        self.ping_x = 0
        self.ping_y = 0
        self.conf_x = 0
        self.conf_y = 0

    def listener_callback(self, msg):
        if len(msg.data) >= 4:  
            self.ping_x = msg.data[0]
            self.conf_x = msg.data[1]
            self.ping_y = msg.data[2]
            self.conf_y = msg.data[3]
        else:
            self.get_logger().warn('Received malformed data!')

class YoloPub(Node):
    def __init__(self):
        super().__init__("yellow_flare_node")
        self.declare_parameter("topic", "command")
        topic_name = self.get_parameter("topic").get_parameter_value().string_value
        self.publisher = self.create_publisher(Int32, topic_name, 10)

    def publish_detection(self, detection_msg):
        msg = Int32()
        msg.data = detection_msg
        self.publisher.publish(msg)
        self.get_logger().info(f'Published: "{msg.data}"')

def main():
    rclpy.init()
    sonar_sub = SonarSubscriber()
    command_pub = YoloPub()
    rclpy.spin_once(sonar_sub, timeout_sec=1)

    global flag_dir_x,flag_dir_y
    lock_y, lock_x = 19.52 ,40#20.25, 38.5
    flag_ping_x = False
    flag_ping_y = False
    flag_bf = False
    flag_dir_x = 0
    flag_dir_y = 0

    FORWARD, LEFT, RIGHT, BACKWARD, STOP, DEPTH_DOWN = 1, 2, 3, 4, 5, 6
    command_pub.publish_detection(93)

    time.sleep(1)
    command_pub.publish_detection(DEPTH_DOWN)
    time.sleep(1)
    command_pub.publish_detection(DEPTH_DOWN)
    time.sleep(0.5)
    command_pub.publish_detection(DEPTH_DOWN)
    time.sleep(0.5)
    command_pub.publish_detection(DEPTH_DOWN)

    def x_corr_locking(lock_x):
        global flag_dir_x
        error_x = sonar_sub.ping_x - lock_x
        if error_x > 0.2 and sonar_sub.conf_x > 90: 
            command_pub.publish_detection(LEFT)
            flag_dir_x = 1
            return False
        elif error_x < -0.2 and sonar_sub.conf_x > 90:
            command_pub.publish_detection(RIGHT)
            flag_dir_x = 2
            return False
        else:
            if flag_dir_x == 1:
                command_pub.publish_detection(RIGHT)
                flag_dir_x = 0 
                return False
            elif flag_dir_x == 2:
                command_pub.publish_detection(LEFT)
                flag_dir_x = 0
                return False
            else:
                return True

    def y_corr_locking(lock_y):
        global flag_dir_y
        error_y = sonar_sub.ping_y - lock_y
        if error_y > 0.2 and sonar_sub.conf_y > 90: 
            command_pub.publish_detection(BACKWARD)
            flag_dir_y = 1
            return False
        elif error_y < -0.2 and sonar_sub.conf_y > 90:
            command_pub.publish_detection(FORWARD)
            flag_dir_y = 2
            return False
        else:
            if flag_dir_y == 1:
                command_pub.publish_detection(FORWARD)
                flag_dir_y = 0 
                return False
            elif flag_dir_y == 2:
                command_pub.publish_detection(BACKWARD)
                flag_dir_y = 0
                return False
            else:
                command_pub.publish_detection(STOP)
                return True

    while rclpy.ok():
        rclpy.spin_once(sonar_sub, timeout_sec=0.1)
        print(sonar_sub.ping_x,sonar_sub.ping_y,sonar_sub.conf_x,sonar_sub.conf_y)
        if not flag_bf:
            flag_ping_x = x_corr_locking(lock_x)
            if flag_ping_x:
                flag_ping_y = y_corr_locking(lock_y)
                if flag_ping_y:
                    flag_bf = True

        if flag_bf:
            command_pub.publish_detection(FORWARD)
            time.sleep(8)
            command_pub.publish_detection(RIGHT)
            time.sleep(8)
            command_pub.publish_detection(BACKWARD)
            time.sleep(16)
            command_pub.publish_detection(LEFT)
            time.sleep(16)
            command_pub.publish_detection(FORWARD)
            time.sleep(16)
            command_pub.publish_detection(RIGHT)
            time.sleep(8)
            command_pub.publish_detection(STOP)
            command_pub.destroy_node()
            sonar_sub.destroy_node()
            rclpy.shutdown()
            break

if __name__ == '__main__':
    main()
