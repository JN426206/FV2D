class DetectedObject():
        
    def __init__(self, bbox, object_class, score, color = None, isout = False, mask = None, trackId = None, pitchxy = None):
        """
        :param bbox:
        :param object_class:
        :param score:
        :param color:
        :param isout:
        :param mask:
        :param trackId:
        :param pitch: Store x,y coordinates from homography pitch
        """
        self.bbox = bbox
        self.object_class = object_class
        self.score = score
        self.color = color
        self.isout = isout
        self.mask = mask
        self.trackId = trackId
        self.pitchxy = pitchxy
        
    def get_xcycwh(self):
        return [int(self.bbox[0]+(self.bbox[2]-self.bbox[0])/2), int(self.bbox[1]+(self.bbox[3]-self.bbox[1])/2), self.bbox[2]-self.bbox[0], self.bbox[3]-self.bbox[1]]
    
    def get_xtlytlwh(self):
        return [self.bbox[0], self.bbox[1], self.bbox[2]-self.bbox[0], self.bbox[3]-self.bbox[1]]
