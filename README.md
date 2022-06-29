# Fixed Camera Object Tracker
Use fixed cameras to track people and cars and then project them onto a building map (targeted for home security).

Works best with a "videos" directory full of MPEG security camera footage.

Uses a combination of Neural Networks (coco) to identify people, cars, and dogs every 10 frames, then uses dlib's image tracking to guess positions during the other frames.

## Intention
This is a fun passion project to refresh my understanding of neural networks, object correlation tools and video processing.  Ideally, I can add into my home cameras and automation system as well.  Possibly even release it for others to use...

# Installation

I suggest using PyCharm as a development environment, then creating a venv virtual environment with the project directory.
After creating a project compiler, install all libraries by adding in settings, or 'pip install cmake', etc.
Install cmake before dlib as dlib cross compiles a bunch of C++ code into being usable by Python

## Windows 10/11 Installation
If building on Windows 10/11, installing Dlib is quite painful:
- revised from: [Install Instructions](https://medium.com/analytics-vidhya/how-to-install-dlib-library-for-python-in-windows-10-57348ba1117f#:~:text=First%20of%20all%2C%20you%20need%20to%20install%20CMake%20library.&text=Then%2C%20you%20can%20install%20dlib%20library%20using%20pip%20install%20).
- From cmake.org/download, install cmake (install CMake to path for all user)
- install "Microsoft VS studio"
- choose the C++ settings (check all the ones listed in [MS List](https://docs.microsoft.com/en-us/cpp/build/cmake-projects-in-visual-studio?view=msvc-170), the Build tools, and C++ CMake settings
- Might need to add CMake to System Path, as described in Install Instructions above
- Reboot
- (Verify that cmake is in your path))
- Open VS Studio, and build one project in Visual Studio using CMake (choose a sample C++ CMake project).  This will set up path and other needed settings.
- Then, go back to PyCharm, and you can now
- pip install cmake
- install dlib (use the GUI or pip install dlib)
- TODO: Determine if all the dlib actions can be done using OpenCV instead

## After all libraries including dlib are installed:
Next, create a configuration to point to "access_cam.py" as the main file for processing
Create a `videos` directory, and put a few mpeg videos with people walking/cars driving in there.

You can now debug or run the program from pycharm, or from python, run
` python access_cam.py ` to run the program

# Options
` python access_cam.py `

` python access_cam.py --cconfig config.yaml`

` python access_cam.py --stream rtsp://admin:PW@@192.168.1.42/Streaming/Channels/102` (to pass in an RTSP live camera stram)

## Optional config.yaml settings
```
tracker:
  settings:
    min_nn_confidence: 0.3 # Show matches that are 30% confident or higher
    nms_threshold: 0.3  # How many overlapping items to suppresss
    apply_ai_every: 9  # Run the computationally expensive AI module every N frames
    distance_before_new_object: 200  # How far away an object has to move from last sighting before it must be a new object
    stream_url: rtsp://admin:PW@@192.168.1.42/Streaming/Channels/102  # URL to rtsp camera
    video_directory: videos  # Directory of local videos to parse (if not using a stream
    
  detectors: # These are the names of layers the Neural Net will look for
    person:
      box_width: 4   # Settings for when 'b' is pressed
      line_width: 2   # Settings for when 'l' is pressed
      color: red
      reduce_box: True # When predicting movement, make the bounding box smaller to better target object
    car:
      box_width: 6
      line_width: 4
      color: blue
    dog:
      ...
```

# Key commands of viewer

```'n' - Next video
'b' - show boxes around targets
'l' - show centroid lines
'p' or ' ' - pause until 'p' or ' ' is pushed again
'j' - write out to JSON current centroid paths
'q' to quit viewer
```

# Feature and to-do List
- Goal - send a message "2 people are walking to the front door"
- Send X,Y paths every frame of where things are seen (along with confidence) to show on a real-time map
- [x] Work with multiple local videos
- [x] Export JSON for post-processing
- [ ] Use multiple types of Neural Networks
- [ ] Use a neural network to train path prediction
- [ ] Learn new faces and accumulate all paths that are typically walked across all cameras to find common routes
- [ ] Integrate into Home Assistant (likely through AppDaemon)
- [ ] Better predict paths and reduce duplicates
- [ ] Send MQTT messages of where items are seen, after mapping to 2d xy space
- [ ] Better count objects along with where they are heading to

