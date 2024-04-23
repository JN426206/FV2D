import numpy as np
import cv2
import colorsys
from sklearn.cluster import KMeans
from collections import Counter

"""
TODO: Wybrać paletę podstawowych kolorów i szukać zawodników dopasowywyjących się do tych kolorów. Kolor drózyny dobrać z k-means i znaleźć najbliższy kolor.
"""

class TeamDetector:
    teamAcolor = None
    teamBcolor = None
    
    once = 0
    main_colors = []
    color_team1 = []
    color_team2 = []
    color_other = []

    def assignTeam(self, img, detectedObject):
        adjust_array = np.array([15, 15, 40])
        
        lower_team1 = np.subtract(TeamDetector.color_team1, adjust_array)
        upper_team1 = np.add(TeamDetector.color_team1, adjust_array)

        lower_team2 = np.subtract(TeamDetector.color_team2, adjust_array)
        upper_team2 = np.add(TeamDetector.color_team2, adjust_array)

        lower_other = np.subtract(TeamDetector.color_other, adjust_array)
        upper_other = np.add(TeamDetector.color_other, adjust_array)

        player_color = self.detectPlayerColor(img, detectedObject)

        mask_team1 = cv2.inRange(player_color, lower_team1, upper_team1)
        mask_team2 = cv2.inRange(player_color, lower_team2, upper_team2)
        mask_other = cv2.inRange(player_color, lower_other, upper_other)
        nonZero1 = cv2.countNonZero(mask_team1)
        nonZero2 = cv2.countNonZero(mask_team2)
        nonZero3 = cv2.countNonZero(mask_other)
        
        maxNonZero = max(nonZero1, nonZero2, nonZero3)
        
        if maxNonZero == nonZero1:
            # self.team = "Team_1"
            return cv2.cvtColor(np.uint8([[TeamDetector.color_team1]]), cv2.COLOR_HSV2BGR).flatten()
        elif maxNonZero == nonZero2:
            # self.team = "Team_2"
            return cv2.cvtColor(np.uint8([[TeamDetector.color_team2]]), cv2.COLOR_HSV2BGR).flatten()
        else:
            # self.team = "Other"
            return cv2.cvtColor(np.uint8([[TeamDetector.color_other]]), cv2.COLOR_HSV2BGR).flatten()
          
    def detectMainColors(self, image, deteced_objects, PERSON_OBJECT_CLASS = 0):
        if TeamDetector.once == 0:
            player_color_list = []
            for detectedObject in deteced_objects:
                if(detectedObject.object_class == PERSON_OBJECT_CLASS and not detectedObject.isout):
                    playerColor = self.detectPlayerColor(image, detectedObject)                
                    player_color_list.append(playerColor)
                    
            # Stack player_color_list arrays in sequence vertically        
            color_matrix = np.vstack((player_color_list))
            # Search for three color group centers (teamAcolor, teamBcolor, otherColor)
            clt = KMeans(n_clusters=3)
            clt.fit(color_matrix)
            n_players = len(clt.labels_)
            # Count number of players in same group (cluster)
            counter = Counter(clt.labels_)
            # Calculation of the percentage share of each cluster
            perc = {}
            for i in counter:
                perc[i] = np.round(counter[i]/n_players, 2)
            # perc = dict(sorted(perc.items()))            
            # Sort from lowest percentage to highest
            perc = dict(sorted(perc.items(), key=lambda item: item[1]))
            # Set center color for teamA, teamB and other 
            TeamDetector.main_colors = clt.cluster_centers_
            
            max_value = list(perc.keys())[2]
            TeamDetector.color_team1 = TeamDetector.main_colors[max_value]
            
            med_value = list(perc.keys())[1]
            TeamDetector.color_team2 = TeamDetector.main_colors[med_value]
            
            min_value = list(perc.keys())[0]
            TeamDetector.color_other = TeamDetector.main_colors[min_value]
            # Lock new detection of teams color
            TeamDetector.once = 1
            # print(main_colors, perc)
            # return main_colors, max_value, med_value, min_value
    
    def first_detect(self, img, x1, x2, y1, y2):
        pass
    
    def k_means(self, img):
        clt = KMeans(n_clusters=4)
        # Reshape image from 3 dimension (x-pixel, y-pixel, color) to 2 dimension (pixel, color)
        # reshapedd = img.reshape(-1, 3)
        clt = clt.fit(img)
        n_pixels = len(clt.labels_)
        # Count number of pixels in same group (cluster)
        counter = Counter(clt.labels_)
        # Calculation of the percentage share of each cluster
        perc = {}
        for i in counter:
            perc[i] = np.round(counter[i]/n_pixels, 2)
        # perc = dict(sorted(perc.items()))
        # Sort from lowest percentage to highest
        perc = dict(sorted(perc.items(), key=lambda item: item[1]))
        # Return percentage share of each cluster and value (in HSV) of this cluster
        return perc, clt.cluster_centers_
    
    def detectPlayerColor(self, img, detectedObject):
        x1, x2, y1, y2 = detectedObject.bbox[0], detectedObject.bbox[2], detectedObject.bbox[1], detectedObject.bbox[3]

        if detectedObject.mask is not None:
            # Keep only the player in the picture
            img = cv2.bitwise_and(img, img, mask = detectedObject.mask)
            # crop_mask = detectedObject.mask[y1:y2, x1:x2]
            # qrt_mask = crop_mask[int(height/6):int(height/2), int(width/5):int(width/1.25)]
        crop = img[y1:y2, x1:x2]
        height, width, channels = crop.shape
        # Crop the image of player to his torso according to ratio        
        if detectedObject.mask is not None:
            qrt = crop[int(height/6):int(height/2), :] 
        else:
            qrt = crop[int(height/6):int(height/2), int(width/5):int(width/1.25)]             
        # cv2.imshow("crop", crop)
        # cv2.imshow("qrt", qrt)
        # cv2.waitKey()
        hsv = cv2.cvtColor(qrt,cv2.COLOR_BGR2HSV)
        
        # Ommit masking green pixels of grass (sports field) if using mask        
        if detectedObject.mask is None:
            # Masking green pixels of grass (sports field)
            mask_green = cv2.inRange(hsv, (33, 25, 102), (84, 255, 161))
            # Cutting out the playing field
            ex_green = cv2.bitwise_and(hsv, hsv, mask=mask_green)
            hsv = hsv-ex_green
        
        clear_hsv = []
        # Remove zero pixels array in hsv
        for pixx in hsv:
            pixxt = []
            for pixy in pixx:
                if pixy.max() != 0:
                    clear_hsv.append(np.array(pixy))
        clear_hsv = np.array(clear_hsv)
            
        # Dominant color search
        perc, colors = self.k_means(clear_hsv)
        
        # max_value = max(perc, key=perc.get)
        # med_temp = list(sorted(perc.values()))[-2]
        # med_value = list(perc.keys())[list(perc.values()).index(med_temp)]
        
        # Perc is sorted in ascending order so 3 = dominant, 0 = weakness
        max_value = list(perc.keys())[3]
        med_value = list(perc.keys())[2]
        # if np.any(np.around(colors[max_value]) <= 0):
        #     return np.around(colors[med_value])
        # else:
        
        # Returns the rounded value to the decimal number of the dominant color in HSV
        return np.around(colors[max_value])