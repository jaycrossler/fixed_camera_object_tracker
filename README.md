# Fixed Camera Object Tracker
Use fixed cameras to track people and cars and then project them onto a building map (targeted for home security)

Works best with a "videos" directory full of MPEG security camera footage

Uses a combination of Neural Networks (coco) to identify people, cars, and dogs every 10 frames, then uses dlib's image tracking to guess positions during the other frames
