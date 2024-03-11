cmake_minimum_required(VERSION 2.8.3)
project(ASR_tests)

set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_BUILD_TYPE RelWithDebInfo)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy 
  std_msgs
  message_generation
)

add_message_files(
  FILES
  Tokens.msg
)

add_service_files(
  FILES
  Classify.srv
  Context.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  LIBRARIES gesture_recognition_action_server
  CATKIN_DEPENDS rospy message_generation
)




## change for being able to use it in Ubuntu 20.04
catkin_install_python(PROGRAMS listen.py my_utils.py classify.py dataset.py context_net.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})


