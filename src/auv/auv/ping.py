import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from brping import Ping1D

class SonarPublisher(Node):
    def __init__(self):
        super().__init__('dual_sonar_publisher')
        # self.conf2 = 100
        self.data_publisher = self.create_publisher(Float32MultiArray, 'ping_topic', 10)
        self.ping_x = Ping1D()
        self.ping_x.connect_serial("/dev/ttyUSB0", 115200)
        self.ping_y = Ping1D()
        self.ping_y.connect_serial("/dev/ttyUSB1", 115200)  # Replace with your second device's port

        if not self.ping_x.initialize():
            self.get_logger().error('Failed to initialize Ping device 1!')
            exit(1)
        if not self.ping_y.initialize():
            self.get_logger().error('Failed to initialize Ping device 2!')
            exit(1)

        self.get_logger().info('Both Ping devices initialized successfully.')

        self.ping_x.set_gain_setting(5)
        self.ping_y.set_gain_setting(3)
        self.ping_x.set_range(500,30000)
        self.ping_y.set_range(500,30000)
        self.ping_x.set_ping_interval(1)
        self.ping_y.set_ping_interval(1)

        self.timer = self.create_timer(0.1, self.publish_data)  # 10 Hz

    def publish_data(self):
        data1 = self.ping_x.get_distance()
        data2 = self.ping_y.get_distance()
        
        array_msg = Float32MultiArray()
        array_msg.data = [
                float(50 - data1['distance'] / 1000.0),  #orient = 5.2,koba=7.2
                float(data1['confidence']),         
                float((25 - data2['distance'] / 1000.0)), #orient = 13,koba=13.3
                float(data2['confidence']  )       
        ]
        self.data_publisher.publish(array_msg)

        self.get_logger().info(
            f"Published Array: Pinger 1 - Distance: {array_msg.data[0]} m, Confidence: {array_msg.data[1]}%, "
            f"Pinger 2 - Distance: {array_msg.data[2]} m, Confidence: {array_msg.data[3]}%")

def main():
    rclpy.init()
    dual_sonar_publisher = SonarPublisher()
    rclpy.spin(dual_sonar_publisher)
    dual_sonar_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
