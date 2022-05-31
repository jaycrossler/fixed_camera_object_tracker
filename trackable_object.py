import colorhash


class TrackableObject:
    centroids = []
    label = None
    color = None
    objectID = None
    last_rect = None

    def __init__(self, objectID, centroid, label=label, color=None, last_rect=None):
        # store the object ID, then initialize a list of centroids

        # using the current centroid
        self.objectID = objectID
        self.centroids = [centroid]
        self.label = label if label else "Object {}".format(objectID)
        self.color = color if color else colorhash.ColorHash(label).rgb
        self.last_rect = last_rect if last_rect else None

        # initialize a boolean used to indicate if the object has
        # already been counted or not
        self.counted = False
