g++ -std=c++11 -O3  -o darknet_detector_test darknet_detector_test.cpp -I../lib_detector -I/home/pi/local_install/include -L../lib_detector -L/home/pi/local_install/lib -L../../darknet_Alexey -ldetector -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio -fopenmp -pthread -lgomp -ldarknet -DOPENCV
rm libdarknet.so
rm libdetector.so
ln -s ../lib_detector/libdetector.so libdetector.so
ln -s ../../darknet_Alexey/libdarknet.so libdarknet.so
