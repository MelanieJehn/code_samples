import os
import sys
file_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(f"{file_path}/wav2vec2-live"))
from live_asr import LiveWav2Vec2
import my_utils
import rospy
from std_msgs.msg import String
from ASR_tests.msg import Tokens
from ASR_tests.srv import *

if __name__ == '__main__':
	#'''
    #    Continuously listen to the microphone until the process is terminated
    #'''

	english_model = "facebook/wav2vec2-large-960h-lv60-self"

	# Change mic_name if not using the KLIM TAlk microphone
	mic_name = 'KLIM Talk: USB Audio(hw: 1, 0)'
	#mic_name = 'default'

	# Start ASR
	asr = LiveWav2Vec2(english_model, device_name=mic_name)
	asr.start()

	try:
		pub = rospy.Publisher('tokens', String, queue_size=100)
		rospy.init_node('pub_node', anonymous=True)
		rate = rospy.Rate(10)

		while not rospy.is_shutdown():
			text, sample_length, inference_time = asr.get_last_text()
			print(f"{sample_length:.3f}s"
			  + f"\t{inference_time:.3f}s"
			  + f"\t{text}")

			#tokens = my_utils.to_token_vec(text, 'distilbert-base-uncased')
			tokens = my_utils.to_token_vec(text, 'bert-base-uncased')
			pub.publish(str(tokens))
			rospy.loginfo(tokens)
			rate.sleep()
		asr.stop()


	except KeyboardInterrupt:
		asr.stop()

	except rospy.ROSInterruptException:
		asr.stop()
		print('Problem occurred with ROS!')

	except rospy.ServiceException as e:
		asr.stop()
		print("Service call failed: %s"%e)
	  
