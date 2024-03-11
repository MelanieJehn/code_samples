
import rospy
from std_msgs.msg import String
from ASR_tests.msg import Tokens
from ASR_tests.srv import *

# A client to test the Classify server

def classify_client():
    rospy.wait_for_service('classify')
    try:
        add_two_ints = rospy.ServiceProxy('classify', Classify)
        resp1 = add_two_ints('hello')
        print(resp1)
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)


if __name__ == "__main__":
    classify_client()