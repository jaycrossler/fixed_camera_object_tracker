import colorhash
import math

TEST_DEAD_RECKONING = True


class TrackableObject:
    centroids = []
    label = None
    color = None
    objectID = None
    last_rect = None
    frame_width = None
    frame_height = None
    detector_type = None
    default_color = None
    color_specifically_set = False

    def __init__(self, objectID, centroid, label=label, color=None, default_color=None,
                 detector_type="Person", last_rect=None, w=2000, h=2000):
        # store the object ID, then initialize a list of centroids

        # using the current centroid
        self.objectID = objectID
        self.centroids = [centroid]
        self.label = label if label else "{} {}".format(detector_type, objectID)
        self.color_specifically_set = True if color else False
        self.color = color if color else colorhash.ColorHash(label).rgb
        self.default_color = default_color if default_color else self.color
        self.last_rect = last_rect if last_rect else None
        self.frame_width = w
        self.frame_height = h
        self.last_detected_center = None
        self.lost_mode = False
        self.detector_type = detector_type

        # initialize a boolean used to indicate if the object has
        # already been counted or not
        self.counted = False

    def set_label(self, label):
        self.label = label
        if not self.color_specifically_set:
            self.color = colorhash.ColorHash(label).rgb

    def add_centroid(self, centroid):
        if TEST_DEAD_RECKONING:
            backtrack = 12
            if self.lost_mode:
                if do_points_match(centroid, self.last_detected_center):
                    # Still lost, keep projecting new path
                    centroid = predict_center_from_last(self.centroids[-(backtrack+2):], num=backtrack,
                                                        x_max=self.frame_width, y_max=self.frame_height)
                else:
                    # Came out of lost mode with a new point
                    self.lost_mode = False
                    self.last_detected_center = centroid
            else:
                last_c = self.centroids[len(self.centroids) - 1]
                last_c2 = self.centroids[len(self.centroids) - 2]
                if do_points_match(centroid, last_c) and do_points_match(centroid, last_c2) and len(self.centroids) > backtrack+3:
                    # Three measures in a row means it likely lost the tracking, go into lost mode
                    self.lost_mode = True

                    # Guess where the center should be from a moving average
                    centroid = predict_center_from_last(self.centroids[-(backtrack+2):-2], num=backtrack,
                                                        x_max=self.frame_width, y_max=self.frame_height)
                else:
                    self.last_detected_center = centroid
            if centroid is None:
                pass
        self.centroids.append(centroid)

# -Math helpers-----------------------------------------------------


def do_points_match(p1, p2):
    # check if the int versions of the first two fields (x and y) match
    return int(p1[0]) == int(p2[0]) and int(p1[1]) == int(p2[1])


def midpoint(p1, p2):
    mx = (p1.getX() + p2.getX()) / 2.0
    my = (p1.getY() + p2.getY()) / 2.0
    return mx, my


def point_dist_from_two_points(p1, p2, dist_mult):
    # TODO: Review this math - should be likely further out
    dist_12 = math.dist(p1, p2)
    if dist_12 > 0:
        # t = dist_mult * dist_12
        try:
            x = p1[0] + (p2[0]-p1[0]) * dist_mult
            y = p1[1] + (p2[1]-p1[1]) * dist_mult
        except IndexError:
            return p1
        #x = ((1-t) * p1[0]) + (t * p2[0])
        #y = ((1-t) * p1[1]) + (t * p2[1])
        return [x, y]
    else:
        return p1


def bound(low, high, value):
    return max(low, min(high, value))


def bound_point(x_max, y_max, pt, x_min=0, y_min=0):
    new_x = bound(x_min, x_max, pt[0])
    new_y = bound(y_min, y_max, pt[1])
    return [new_x, new_y]


def predict_center_from_last(point_list, num=None, x_max=2000, y_max=2000):
    if num is None:
        num = len(point_list) - 1

    sub_list = point_list[-num:]

    predictions = []
    recent_point = sub_list[-1]
    for i, past_point in enumerate(sub_list):
        if i < len(sub_list)-1:
            next_point = sub_list[i+1]
            # if past_point == next_point:
            #     continue
            predicted_point = point_dist_from_two_points(past_point, recent_point, (num+1-i)/(num-i))
            predicted_point = bound_point(x_max, y_max, predicted_point)

            predictions.append(predicted_point)

    average = [int(sum(x) / len(x)) for x in zip(*predictions)]

    return average
