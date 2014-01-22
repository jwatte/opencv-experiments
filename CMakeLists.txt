cmake_minimum_required(VERSION 2.8)
project(OpenCV-Experiments)
find_package(OpenCV REQUIRED)
ADD_DEFINITIONS(
    -std=c++11
)

SET(RASPICAM_CV $ENV{HOME}/dev/robidouille/raspicam_cv)
SET(USERLAND $ENV{HOME}/dev/userland)
include_directories(${RASPICAM_CV})

SET(MMAL_LIBS
    ${USERLAND}/build/lib/libmmal_core.so
    ${USERLAND}/build/lib/libmmal_util.so
    ${USERLAND}/build/lib/libmmal_vc_client.so
    ${USERLAND}/build/lib/libvcos.so)
SET(BCM_LIBS ${USERLAND}/build/lib/libbcm_host.so)
SET(GLES_LIBS ${USERLAND}/build/lib/libGLESv2.so)
SET(EGL_LIBS ${USERLAND}/build/lib/libEGL.so)
SET(DISPLAY_CAMERA_LIBS
    ${RASPICAM_CV}/libraspicamcv.a
    ${MMAL_LIBS} ${BCM_LIBS} ${OpenCV_LIBS})

# Pi executables
#SET(DISPLAY_IMAGE build/display-image)
#SET(DISPLAY_VIDEO build/display-video)
#SET(DISPLAY_CAMERA build/display-camera)
#SET(PROFILE_CAMERA build/profile-camera)
#add_executable(${DISPLAY_IMAGE} src/display-image.cpp)
#add_executable(${DISPLAY_VIDEO} src/display-video.cpp)
#add_executable(${DISPLAY_CAMERA} src/display-camera.cpp)
#add_executable(${PROFILE_CAMERA} src/profile-camera.cpp)
#target_link_libraries(${DISPLAY_IMAGE} ${OpenCV_LIBS})
#target_link_libraries(${DISPLAY_VIDEO} ${OpenCV_LIBS})
#target_link_libraries(${DISPLAY_CAMERA} ${DISPLAY_CAMERA_LIBS})
#target_link_libraries(${PROFILE_CAMERA} ${DISPLAY_CAMERA_LIBS})

# Platform independent
SET(OPTICAL_FLOW build/optical-flow)
SET(OCV_EXAMPLE build/farneback-example)
add_executable(${OPTICAL_FLOW} src/dense-sfm.cpp)
add_executable(${OCV_EXAMPLE} src/fback.cpp)
target_link_libraries(${OPTICAL_FLOW} ${OpenCV_LIBS})
target_link_libraries(${OCV_EXAMPLE} ${OpenCV_LIBS})
