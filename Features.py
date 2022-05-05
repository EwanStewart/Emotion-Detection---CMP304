import math

class Features:

    def __init__(self, landmarks):
        self.landmarks = landmarks
        self.f_vector = []

        
    def run(self):
        
        self.calculateLeftEyeBrowLength()
        self.calculateRightEyeBrowLength()
        self.calculateLipWidth()
        self.calculateLipHeight()
        self.calculateLeftEyeHeight()
        self.calculateRightEyeHeight()
        self.calculateLeftEyeWidth()
        self.calculateRightEyeWidth()

        return self.f_vector


    def calculateLeftEyeBrowLength(self):
        l_eyebrow_left = self.landmarks.part(17).x
        l_eyebrow_right = self.landmarks.part(21).x
        l_eyebrow_length = l_eyebrow_right - l_eyebrow_left
        l_eyebrow_length = l_eyebrow_length / (self.landmarks.part(16).x - self.landmarks.part(0).x)
        self.f_vector.append(l_eyebrow_length)

    def calculateRightEyeBrowLength(self):
        r_eyebrow_left = self.landmarks.part(22).x
        r_eyebrow_right = self.landmarks.part(26).x
        r_eyebrow_length = r_eyebrow_right - r_eyebrow_left
        r_eyebrow_length = r_eyebrow_length / (self.landmarks.part(16).x - self.landmarks.part(0).x)
        self.f_vector.append(r_eyebrow_length)

    def calculateLipWidth(self):
        lip_left = self.landmarks.part(48).x
        lip_right = self.landmarks.part(54).x
        lip_width = lip_right - lip_left
        lip_width = lip_width / (self.landmarks.part(16).x - self.landmarks.part(0).x)
        self.f_vector.append(lip_width)

    def calculateLipHeight(self):
        lip_top = self.landmarks.part(51).y
        lip_bottom = self.landmarks.part(57).y
        lip_height = lip_bottom - lip_top
        lip_height = lip_height / (self.landmarks.part(16).x - self.landmarks.part(0).x)
        self.f_vector.append(lip_height)

    def calculateLeftEyeHeight(self):
        l_eye_top = self.landmarks.part(37).y
        l_eye_bottom = self.landmarks.part(41).y
        l_eye_height = l_eye_bottom - l_eye_top
        l_eye_height = l_eye_height / (self.landmarks.part(16).x - self.landmarks.part(0).x)
        self.f_vector.append(l_eye_height)

    def calculateRightEyeHeight(self):
        r_eye_top = self.landmarks.part(43).y
        r_eye_bottom = self.landmarks.part(47).y
        r_eye_height = r_eye_bottom - r_eye_top
        r_eye_height = r_eye_height / (self.landmarks.part(16).x - self.landmarks.part(0).x)
        self.f_vector.append(r_eye_height)

    def calculateLeftEyeWidth(self):
        l_eye_left = self.landmarks.part(36).x
        l_eye_right = self.landmarks.part(39).x
        l_eye_width = l_eye_right - l_eye_left
        l_eye_width = l_eye_width / (self.landmarks.part(16).x - self.landmarks.part(0).x)
        self.f_vector.append(l_eye_width)

    def calculateRightEyeWidth(self):
        r_eye_left = self.landmarks.part(42).x
        r_eye_right = self.landmarks.part(45).x
        r_eye_width = r_eye_right - r_eye_left
        r_eye_width = r_eye_width / (self.landmarks.part(16).x - self.landmarks.part(0).x)
        self.f_vector.append(r_eye_width)

    def calculateNoseLength(self):
        nose_left = self.landmarks.part(27).x
        nose_right = self.landmarks.part(30).x
        nose_length = nose_right - nose_left
        nose_length = nose_length / (self.landmarks.part(16).x - self.landmarks.part(0).x)
        self.f_vector.append(nose_length)





