from datetime import datetime

class SmartHomeLighting:
    def __init__(self, number_of_people):
        self.number_of_people = number_of_people
        self.lights_on = False
        self.lights_dimmed = False

    def get_current_time(self):
        return datetime.now()

    def adjust_lighting(self):
        current_time = self.get_current_time()
        is_night = current_time.hour >= 18 or current_time.hour < 6

        if is_night:
            self.lights_on = True
            self.lights_dimmed = False
        else:
            self.lights_on = True
            self.lights_dimmed = True

        if self.number_of_people == 0:
            self.lights_on = False
            self.lights_dimmed = False

        if current_time.hour >= 23 or current_time.hour < 6:
            self.lights_on = False
            self.lights_dimmed = False

        if 6 <= current_time.hour < 18:
            self.lights_on = True
            self.lights_dimmed = True

        self.display_light_status()

    def display_light_status(self):
        status = "on" if self.lights_on else "off"
        dimmed_status = "dimmed" if self.lights_dimmed else "bright"
        if self.lights_on:
            print(f"The lights are {status} and {dimmed_status}.")
        else:
            print(f"The lights are {status}.")

smart_home_lighting = SmartHomeLighting(number_of_people=2)
smart_home_lighting.adjust_lighting()

