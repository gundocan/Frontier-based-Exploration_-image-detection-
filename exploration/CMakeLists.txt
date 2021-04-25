cmake_minimum_required(VERSION 2.8.3)
project(exploration)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  actionlib_msgs
  message_generation
)

catkin_python_setup()

add_message_files(
  FILES
  Path.msg
  status.msg
)

add_service_files(
  FILES
  GenerateFrontier.srv
  PlanPath.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs  # Or other packages containing msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

install(PROGRAMS
    scripts/explorer.py
    scripts/frontier.py
    scripts/path_follower.py
    scripts/planner.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
