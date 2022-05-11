import math

class Features:

    def __init__(self, landmarks):
        self.landmarks = landmarks
        self.f_vector = []

        
    def run(self):
        
        self.calculateLeftEyeBrowLength()
        self.calculateRightEyeBrowLength()
        self.calculateUpperLeftLipLength()
        self.calculateUpperRightLipLength()
        self.calculateLipLength()
        self.calculateLipHeight()

        return self.f_vector

    def distance(self, p1, p2):
        return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)
    

    def calculateLeftEyeBrowLength(self):
        dist = 0
        totalDist = 0

        for i in range(19, 22):
            dist = self.distance(self.landmarks.part(i), self.landmarks.part(40))
            dist = dist / self.distance(self.landmarks.part(0), self.landmarks.part(16))
            totalDist += dist

        self.f_vector.append(totalDist)

    def calculateRightEyeBrowLength(self):
        dist = 0
        totalDist = 0

        for i in range(23, 26):
            dist = self.distance(self.landmarks.part(i), self.landmarks.part(43))
            dist = dist / self.distance(self.landmarks.part(0), self.landmarks.part(16))
            totalDist += dist

        self.f_vector.append(totalDist)

    def calculateUpperLeftLipLength(self):
        dist = 0
        totalDist = 0

        for i in range(49, 51):
            dist = self.distance(self.landmarks.part(i), self.landmarks.part(34))
            dist = dist / self.distance(self.landmarks.part(0), self.landmarks.part(16))
            totalDist += dist
        
        self.f_vector.append(totalDist)

    def calculateUpperRightLipLength(self):
        dist = 0
        totalDist = 0

        for i in range(53, 55):
            dist = self.distance(self.landmarks.part(i), self.landmarks.part(34))
            dist = dist / self.distance(self.landmarks.part(0), self.landmarks.part(16))
            totalDist += dist
        
        self.f_vector.append(totalDist)

    def calculateLipLength(self):
        dist = 0

        dist = self.distance(self.landmarks.part(49), self.landmarks.part(55))   
        dist = dist / self.distance(self.landmarks.part(0), self.landmarks.part(16))
        self.f_vector.append(dist)             

    def calculateLipHeight(self):
        dist = 0

        dist = self.distance(self.landmarks.part(52), self.landmarks.part(58))   
        dist = dist / self.distance(self.landmarks.part(0), self.landmarks.part(16))
        self.f_vector.append(dist)   


    def landmarksOnly(self, landmarks):
        for n in range(0, 68):
            self.f_vector.append(landmarks.part(n).x)
            self.f_vector.append(landmarks.part(n).y)
        
        return self.f_vector




