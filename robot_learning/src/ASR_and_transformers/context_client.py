import rospy
from std_msgs.msg import String
from ASR_tests.msg import Tokens
from ASR_tests.srv import *

# A client to test the Context server


def classify_client():
    rospy.wait_for_service('get_context')
    try:
        add_two_ints = rospy.ServiceProxy('get_context', Context)
        # Msg: Give me the cup please
        msg = "[101, 2507, 2033, 1996, 2452, 3531, 102]"
        resp1 = add_two_ints(msg)
        print(resp1)
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)


if __name__ == "__main__":
    classify_client()